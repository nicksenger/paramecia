fn main() {
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let wit_dir = out.join("paramecia-wit");
    paramecia_wit::write_wit_to(&wit_dir).expect("failed to write WIT files");

    let bindgen = format!(
        r#"wit_bindgen::generate!({{
    world: "controller",
    path: "{}",
    generate_all,
}});"#,
        wit_dir.display(),
    );
    std::fs::write(out.join("paramecia_bindgen.rs"), bindgen)
        .expect("failed to write bindgen file");
}
