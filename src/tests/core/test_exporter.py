from fenn.core.exporter import Exporter


class TestExporter:
    def test_configure_uses_yaml_export_dir(self, tmp_path):
        exporter = Exporter()

        root_dir = exporter.configure(
            {
                "project": "demo-project",
                "export": {"dir": str(tmp_path / "artifacts")},
            }
        )

        assert root_dir == tmp_path / "artifacts"
        assert exporter.root_dir == tmp_path / "artifacts"
        assert root_dir.exists()

    def test_get_export_dir_includes_project_by_default(self, tmp_path):
        exporter = Exporter()
        exporter.configure(
            {
                "project": "demo-project",
                "export": {"dir": str(tmp_path / "exports")},
            }
        )

        model_dir = exporter.get_export_dir("sequence_classifier")

        assert model_dir == tmp_path / "exports" / "demo-project" / "sequence_classifier"
        assert model_dir.exists()

    def test_get_export_dir_can_skip_project_prefix(self, tmp_path):
        exporter = Exporter()
        exporter.configure(
            {
                "project": "demo-project",
                "export": {"dir": str(tmp_path / "exports")},
            }
        )

        export_dir = exporter.get_export_dir("shared", include_project=False)

        assert export_dir == tmp_path / "exports" / "shared"
        assert export_dir.exists()