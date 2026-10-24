// Compiles the protobuf contracts in proto/ with protox (pure Rust), so neither
// this crate nor anything depending on it needs protoc installed.

const PROTOS: &[&str] = &[
    "skylens/model/v1/model.proto",
    "skylens/viewer/v1/viewer.proto",
    "skylens/payload/v1/payload.proto",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=proto");
    let grpc = std::env::var_os("CARGO_FEATURE_GRPC").is_some();
    let fds = protox::compile(PROTOS, ["proto"])?;
    tonic_prost_build::configure()
        .build_client(grpc)
        .build_server(grpc)
        .compile_fds(fds)?;
    Ok(())
}
