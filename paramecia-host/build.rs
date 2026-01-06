fn main() {
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let wit_dir = out.join("paramecia-wit");
    paramecia_wit::write_wit_to(&wit_dir).expect("failed to write WIT files");

    let bindgen = format!(
        r#"wasmtime::component::bindgen!({{
    path: "{}",
    world: "controller",
    imports: {{
        "paramecia:host/structure": async | trappable,
        "paramecia:host/structure-ext": async | trappable,
        "paramecia:host/inference": async | trappable,
        "paramecia:host/inference-ext": async | trappable,
        "paramecia:host/training": async | trappable,
        "paramecia:host/training-ext": async | trappable,
        "paramecia:host/interactive": async | trappable,
    }},
    exports: {{
        default: async | store | task_exit,
    }},
    with: {{
        "wasi": wasmtime_wasi::p3::bindings,
    }},
}});"#,
        wit_dir.display(),
    );
    std::fs::write(out.join("paramecia_bindgen.rs"), bindgen)
        .expect("failed to write bindgen file");
}
