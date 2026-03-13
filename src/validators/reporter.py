from pathlib import Path

from src.domain_models.dtos import ValidationReport


def generate_html_report(report: ValidationReport, save_path: Path) -> None:
    """Generates an HTML validation report with metrics."""
    # Build simple HTML string

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Validation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .passed {{ color: green; font-weight: bold; }}
            .failed {{ color: red; font-weight: bold; }}
            table {{ border-collapse: collapse; width: 50%; margin-top: 20px; }}
            th, td {{ border: 1px solid #dddddd; text-align: left; padding: 8px; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Validation Report</h1>
        <p>Overall Status: <span class="{"passed" if report.passed else "failed"}">{"PASS" if report.passed else "FAIL"}</span></p>
        {f"<p>Reason: {report.reason}</p>" if report.reason else ""}

        <h2>Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Energy RMSE</td>
                <td>{report.energy_rmse:.5f} eV/atom</td>
            </tr>
            <tr>
                <td>Force RMSE</td>
                <td>{report.force_rmse:.5f} eV/A</td>
            </tr>
            <tr>
                <td>Stress RMSE</td>
                <td>{report.stress_rmse:.5f} GPa</td>
            </tr>
            <tr>
                <td>Phonon Stable</td>
                <td>{"Yes" if report.phonon_stable else "No"}</td>
            </tr>
            <tr>
                <td>Mechanically Stable</td>
                <td>{"Yes" if report.mechanically_stable else "No"}</td>
            </tr>
        </table>

        <h2>Visualizations</h2>
        <p>Parity Plots and Phonon bands will be linked here when generated.</p>
    </body>
    </html>
    """

    with Path.open(save_path, "w") as f:
        f.write(html_content)
