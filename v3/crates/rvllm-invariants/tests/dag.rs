use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

fn set(names: &[&'static str]) -> BTreeSet<&'static str> {
    names.iter().copied().collect()
}

fn allowed_deps() -> BTreeMap<&'static str, BTreeSet<&'static str>> {
    BTreeMap::from([
        ("rvllm-core", set(&[])),
        ("rvllm-metal", set(&["rvllm-core"])),
        ("rvllm-mem", set(&["rvllm-core", "rvllm-metal"])),
        ("rvllm-kernels", set(&["rvllm-core", "rvllm-mem"])),
        (
            "rvllm-cutlass",
            set(&["rvllm-core", "rvllm-kernels", "rvllm-mem"]),
        ),
        (
            "rvllm-attention",
            set(&["rvllm-core", "rvllm-kernels", "rvllm-mem", "rvllm-metal"]),
        ),
        (
            "rvllm-fused",
            set(&["rvllm-core", "rvllm-kernels", "rvllm-mem"]),
        ),
        ("rvllm-metadata", set(&["rvllm-core", "rvllm-mem"])),
        (
            "rvllm-graph",
            set(&["rvllm-core", "rvllm-mem", "rvllm-metadata"]),
        ),
        (
            "rvllm-loader",
            set(&["rvllm-core", "rvllm-mem", "rvllm-metal"]),
        ),
        (
            "rvllm-sampling",
            set(&["rvllm-core", "rvllm-fused", "rvllm-kernels", "rvllm-mem"]),
        ),
        ("rvllm-vision", set(&[])),
        ("rvllm-imageio", set(&[])),
        (
            "rvllm-runtime",
            set(&[
                "rvllm-attention",
                "rvllm-core",
                "rvllm-cutlass",
                "rvllm-fused",
                "rvllm-graph",
                "rvllm-kernels",
                "rvllm-loader",
                "rvllm-mem",
                "rvllm-metadata",
                "rvllm-metal",
                "rvllm-sampling",
            ]),
        ),
        (
            "rvllm-serve",
            set(&[
                "rvllm-core",
                "rvllm-imageio",
                "rvllm-kernels",
                "rvllm-loader",
                "rvllm-mem",
                "rvllm-metal",
                "rvllm-runtime",
                "rvllm-vision",
            ]),
        ),
        (
            "rvllm-bench",
            set(&[
                "rvllm-core",
                "rvllm-cutlass",
                "rvllm-fused",
                "rvllm-kernels",
                "rvllm-loader",
                "rvllm-mem",
                "rvllm-metal",
                "rvllm-runtime",
            ]),
        ),
        ("rvllm-mcp", set(&[])),
        ("rvllm-invariants", set(&[])),
    ])
}

fn crate_budgets() -> BTreeMap<&'static str, usize> {
    BTreeMap::from([
        ("rvllm-runtime", 16_000),
        ("rvllm-loader", 10_000),
        ("rvllm-metal", 10_000),
        ("rvllm-serve", 4_000),
        ("rvllm-cutlass", 4_000),
        ("rvllm-fused", 3_300),
        ("rvllm-attention", 3_000),
        ("rvllm-mem", 3_000),
        ("rvllm-vision", 2_900),
        ("rvllm-core", 2_200),
        ("rvllm-bench", 2_500),
        ("rvllm-kernels", 1_500),
        ("rvllm-sampling", 1_200),
        ("rvllm-mcp", 800),
        ("rvllm-imageio", 600),
        ("rvllm-metadata", 600),
        ("rvllm-graph", 700),
        ("rvllm-invariants", 400),
    ])
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn manifests() -> BTreeMap<String, PathBuf> {
    let crates = workspace_root().join("crates");
    let mut manifests = BTreeMap::new();
    for entry in fs::read_dir(crates).expect("read crates directory") {
        let path = entry.expect("read crate entry").path();
        let manifest = path.join("Cargo.toml");
        if !manifest.is_file() {
            continue;
        }
        let document = fs::read_to_string(&manifest).expect("read crate manifest");
        let value: toml::Value = toml::from_str(&document).expect("parse crate manifest");
        let name = value["package"]["name"]
            .as_str()
            .expect("package.name")
            .to_string();
        assert!(
            manifests.insert(name.clone(), manifest).is_none(),
            "duplicate package name {name}"
        );
    }
    manifests
}

fn collect_rvllm_deps(value: &toml::Value, out: &mut BTreeSet<String>) {
    let Some(table) = value.as_table() else {
        return;
    };
    for (key, value) in table {
        if matches!(
            key.as_str(),
            "dependencies" | "dev-dependencies" | "build-dependencies"
        ) {
            let deps = value
                .as_table()
                .expect("dependency section must be a table");
            out.extend(
                deps.keys()
                    .filter(|name| name.starts_with("rvllm-"))
                    .cloned(),
            );
        }
        collect_rvllm_deps(value, out);
    }
}

#[test]
fn workspace_members_and_dependency_dag_are_complete() {
    let expected = allowed_deps();
    let manifests = manifests();
    let actual_names: BTreeSet<_> = manifests.keys().map(String::as_str).collect();
    let expected_names: BTreeSet<_> = expected.keys().copied().collect();
    assert_eq!(
        actual_names, expected_names,
        "update the explicit crate DAG"
    );

    let workspace_text =
        fs::read_to_string(workspace_root().join("Cargo.toml")).expect("read workspace manifest");
    let workspace: toml::Value = toml::from_str(&workspace_text).expect("parse workspace manifest");
    let declared: BTreeSet<_> = workspace["workspace"]["members"]
        .as_array()
        .expect("workspace.members")
        .iter()
        .map(|value| value.as_str().expect("member string"))
        .filter_map(|member| member.strip_prefix("crates/"))
        .collect();
    assert_eq!(
        declared, expected_names,
        "workspace member list is incomplete"
    );

    let mut violations = Vec::new();
    for (name, manifest) in &manifests {
        let text = fs::read_to_string(manifest).expect("read crate manifest");
        let value: toml::Value = toml::from_str(&text).expect("parse crate manifest");
        let mut deps = BTreeSet::new();
        collect_rvllm_deps(&value, &mut deps);
        for dep in deps {
            if !expected[name.as_str()].contains(dep.as_str()) {
                violations.push(format!("{name} -> {dep}"));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "rvLLM dependency DAG violations:\n{}",
        violations.join("\n")
    );
}

fn source_files(root: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(root).expect("read source tree") {
        let path = entry.expect("read source entry").path();
        if path.is_dir() {
            if path.file_name().is_some_and(|name| name == "target") {
                continue;
            }
            source_files(&path, out);
            continue;
        }
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| {
                matches!(
                    ext,
                    "c" | "cc" | "cpp" | "cu" | "cuh" | "h" | "hpp" | "metal" | "rs"
                )
            })
        {
            out.push(path);
        }
    }
}

#[test]
fn source_size_budgets_cover_every_crate_and_file() {
    const MAX_FILE_LINES: usize = 6_500;

    let manifests = manifests();
    let budgets = crate_budgets();
    let actual_names: BTreeSet<_> = manifests.keys().map(String::as_str).collect();
    let budget_names: BTreeSet<_> = budgets.keys().copied().collect();
    assert_eq!(
        actual_names, budget_names,
        "every crate needs an explicit budget"
    );

    let mut violations = Vec::new();
    for (name, manifest) in manifests {
        let root = manifest.parent().expect("crate directory");
        let mut files = Vec::new();
        source_files(root, &mut files);
        let mut crate_lines = 0;
        for file in files {
            let lines = fs::read_to_string(&file)
                .unwrap_or_else(|e| panic!("read {}: {e}", file.display()))
                .lines()
                .count();
            crate_lines += lines;
            if lines > MAX_FILE_LINES {
                violations.push(format!(
                    "{} has {lines} lines (file limit {MAX_FILE_LINES})",
                    file.display()
                ));
            }
        }
        if crate_lines > budgets[name.as_str()] {
            violations.push(format!(
                "{name} has {crate_lines} source lines (crate limit {})",
                budgets[name.as_str()]
            ));
        }
    }
    assert!(
        violations.is_empty(),
        "source-size budget violations:\n{}",
        violations.join("\n")
    );
}

#[test]
fn parser_covers_target_and_build_dependency_tables() {
    let value: toml::Value = toml::from_str(
        r#"
        [build-dependencies]
        rvllm-core = { path = "../core" }

        [target.'cfg(target_os = "linux")'.dependencies]
        rvllm-metal = { path = "../metal" }

        [dev-dependencies.rvllm-loader]
        path = "../loader"
        "#,
    )
    .expect("parse fixture");
    let mut deps = BTreeSet::new();
    collect_rvllm_deps(&value, &mut deps);
    assert_eq!(
        deps,
        ["rvllm-core", "rvllm-loader", "rvllm-metal"]
            .into_iter()
            .map(str::to_string)
            .collect()
    );
}
