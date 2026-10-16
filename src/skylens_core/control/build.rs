// Embeds the control tower icon into the Windows executable. Other targets need nothing.

fn main() {
    println!("cargo:rerun-if-changed=../../../res/icon/control/skylens-control.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("../../../res/icon/control/skylens-control.ico");
        if let Err(e) = res.compile() {
            println!("cargo:warning=could not embed the exe icon: {e}");
        }
    }
}
