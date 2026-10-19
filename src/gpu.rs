//! discrete GPUs and how much of their dedicated memory is in use.
//!
//! sizes come from DXGI, whose DedicatedVideoMemory is a SIZE_T and so not
//! capped at 4 GiB the way WMI's Win32_VideoController.AdapterRAM is. usage
//! comes from the "\GPU Adapter Memory(*)\Dedicated Usage" performance
//! counters, whose instances are named after the adapter LUID
//! ("luid_0x00000000_0x000162CD_phys_0"), which is how the two are joined.

use anyhow::{bail, Result};
use std::collections::HashMap;

/// an adapter with less dedicated memory than this is taken for an integrated
/// GPU: an iGPU's "dedicated" memory is only the carve-out the firmware
/// reserves from system RAM, typically 512 MiB
const DISCRETE_MIN_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Microsoft's PCI vendor id, used by the Basic Render Driver and other software adapters
const VENDOR_MICROSOFT: u32 = 0x1414;

pub type Luid = (u32, u32);

#[derive(Clone, Debug)]
pub struct Adapter {
    pub name: String,
    pub luid: Luid,
    pub dedicated: u64,
    /// how much local memory Windows lets a process use on this adapter right now,
    /// from DXGI QueryVideoMemoryInfo; 0 when the adapter does not say
    pub budget: u64,
}

#[cfg(windows)]
pub fn discrete_adapters() -> Vec<Adapter> {
    use windows::core::Interface;
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, IDXGIAdapter3, IDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
        DXGI_QUERY_VIDEO_MEMORY_INFO,
    };

    let mut out = Vec::new();
    let Ok(factory) = (unsafe { CreateDXGIFactory1::<IDXGIFactory1>() }) else {
        return out;
    };
    let mut i = 0;
    while let Ok(adapter) = unsafe { factory.EnumAdapters1(i) } {
        i += 1;
        let Ok(desc) = (unsafe { adapter.GetDesc1() }) else { continue };
        let software = desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0;
        let dedicated = desc.DedicatedVideoMemory as u64;
        if software || desc.VendorId == VENDOR_MICROSOFT || dedicated < DISCRETE_MIN_BYTES {
            continue;
        }
        let len = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
        let luid = (desc.AdapterLuid.HighPart as u32, desc.AdapterLuid.LowPart);
        // linked adapters can show up more than once under the same LUID
        if out.iter().any(|a: &Adapter| a.luid == luid) {
            continue;
        }
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let budget = match adapter.cast::<IDXGIAdapter3>() {
            Ok(a3) => match unsafe { a3.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info) } {
                Ok(()) => info.Budget,
                Err(_) => 0,
            },
            Err(_) => 0,
        };
        out.push(Adapter { name: String::from_utf16_lossy(&desc.Description[..len]), luid, dedicated, budget });
    }
    out
}

#[cfg(not(windows))]
pub fn discrete_adapters() -> Vec<Adapter> {
    Vec::new()
}

/// dedicated VRAM a model can use on these GPUs without Windows moving part of
/// it to system RAM: each GPU's size less the reserve
pub fn usable(adapters: &[Adapter], reserve: u64) -> u64 {
    adapters.iter().map(|a| a.dedicated.saturating_sub(reserve)).sum()
}

/// dedicated VRAM in use on `adapters`, or None when the counters cannot be read
pub fn used(sampler: &mut Option<UsageSampler>, adapters: &[Adapter]) -> Option<u64> {
    let by_luid = sampler.as_mut()?.sample().ok()?;
    Some(adapters.iter().map(|a| by_luid.get(&a.luid).copied().unwrap_or(0)).sum())
}

/// parses "luid_0x00000000_0x000162CD_phys_0" into (high, low)
fn parse_instance(name: &str) -> Option<Luid> {
    let mut parts = name.split('_');
    if parts.next()? != "luid" {
        return None;
    }
    let hex = |s: &str| u32::from_str_radix(s.trim_start_matches("0x").trim_start_matches("0X"), 16).ok();
    let high = hex(parts.next()?)?;
    let low = hex(parts.next()?)?;
    Some((high, low))
}

/// a PDH query over the dedicated-usage counter of every adapter. PDH handles
/// belong to the thread that made them, so a sampler stays on one thread.
pub struct UsageSampler {
    #[cfg(windows)]
    query: isize,
    #[cfg(windows)]
    counter: isize,
}

#[cfg(windows)]
impl UsageSampler {
    pub fn new() -> Result<Self> {
        use windows::core::w;
        use windows::Win32::System::Performance::{PdhAddEnglishCounterW, PdhOpenQueryW};

        let mut query = 0isize;
        let status = unsafe { PdhOpenQueryW(None, 0, &mut query) };
        if status != 0 {
            bail!("PdhOpenQuery failed -- 0x{status:08x}");
        }
        let mut counter = 0isize;
        let status =
            unsafe { PdhAddEnglishCounterW(query, w!("\\GPU Adapter Memory(*)\\Dedicated Usage"), 0, &mut counter) };
        if status != 0 {
            bail!("PdhAddEnglishCounter failed -- 0x{status:08x}");
        }
        Ok(UsageSampler { query, counter })
    }

    /// bytes of dedicated memory in use, per adapter LUID
    pub fn sample(&mut self) -> Result<HashMap<Luid, u64>> {
        use windows::Win32::System::Performance::{
            PdhCollectQueryData, PdhGetFormattedCounterArrayW, PDH_FMT_COUNTERVALUE_ITEM_W, PDH_FMT_LARGE,
            PDH_MORE_DATA,
        };

        let status = unsafe { PdhCollectQueryData(self.query) };
        if status != 0 {
            bail!("PdhCollectQueryData failed -- 0x{status:08x}");
        }
        let mut size = 0u32;
        let mut count = 0u32;
        let status = unsafe { PdhGetFormattedCounterArrayW(self.counter, PDH_FMT_LARGE, &mut size, &mut count, None) };
        if status != PDH_MORE_DATA {
            bail!("PdhGetFormattedCounterArray failed -- 0x{status:08x}");
        }
        // the buffer holds the items followed by the instance name strings they point into
        let item_size = std::mem::size_of::<PDH_FMT_COUNTERVALUE_ITEM_W>();
        let mut buf: Vec<PDH_FMT_COUNTERVALUE_ITEM_W> = Vec::with_capacity((size as usize).div_ceil(item_size));
        let status = unsafe {
            PdhGetFormattedCounterArrayW(self.counter, PDH_FMT_LARGE, &mut size, &mut count, Some(buf.as_mut_ptr()))
        };
        if status != 0 {
            bail!("PdhGetFormattedCounterArray failed -- 0x{status:08x}");
        }
        unsafe { buf.set_len(count as usize) };

        let mut out: HashMap<Luid, u64> = HashMap::new();
        for item in &buf {
            let name = unsafe { item.szName.to_string() }.unwrap_or_default();
            if item.FmtValue.CStatus != 0 {
                continue;
            }
            if let Some(luid) = parse_instance(&name) {
                let v = unsafe { item.FmtValue.Anonymous.largeValue }.max(0) as u64;
                *out.entry(luid).or_default() += v;
            }
        }
        Ok(out)
    }
}

#[cfg(windows)]
impl Drop for UsageSampler {
    fn drop(&mut self) {
        unsafe { windows::Win32::System::Performance::PdhCloseQuery(self.query) };
    }
}

#[cfg(not(windows))]
impl UsageSampler {
    pub fn new() -> Result<Self> {
        bail!("GPU usage counters are only available on Windows")
    }

    pub fn sample(&mut self) -> Result<HashMap<Luid, u64>> {
        Ok(HashMap::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_names() {
        assert_eq!(parse_instance("luid_0x00000000_0x000162CD_phys_0"), Some((0, 0x162cd)));
        assert_eq!(parse_instance("luid_0x00000001_0x0000d1b4_phys_1"), Some((1, 0xd1b4)));
        assert_eq!(parse_instance("pid_1234_luid_0x0_0x1"), None);
    }
}
