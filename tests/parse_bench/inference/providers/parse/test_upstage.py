"""Tests for the Upstage Document Parse provider."""

import io
import json
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from parse_bench.evaluation.layout_adapters import create_layout_adapter_for_result
from parse_bench.evaluation.layout_label_mappers import build_mapping_context, resolve_layout_label_mapper
from parse_bench.inference.pipelines import get_pipeline
from parse_bench.inference.providers.base import (
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse.upstage import UpstageDocumentParseProvider
from parse_bench.schemas.layout_detection_output import LayoutDetectionModel
from parse_bench.schemas.layout_ontology import CanonicalLabel
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult


def element(category: str, markup: str, page: int = 1, coordinates=None) -> dict:
    return {
        "category": category,
        "page": page,
        "content": {"html": markup, "markdown": markup, "text": markup},
        "coordinates": (
            coordinates
            if coordinates is not None
            else [
                {"x": 0.1, "y": 0.2},
                {"x": 0.8, "y": 0.2},
                {"x": 0.8, "y": 0.6},
                {"x": 0.1, "y": 0.6},
            ]
        ),
    }


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> UpstageDocumentParseProvider:
    monkeypatch.setenv("UPSTAGE_API_KEY", "test-key-not-real")
    monkeypatch.delenv("UPSTAGE_BASE_URL", raising=False)
    spec = get_pipeline("upstage_dpe_v2")
    return UpstageDocumentParseProvider("upstage", spec.config)


def raw(body: dict) -> RawInferenceResult:
    spec = get_pipeline("upstage_dpe_v2")
    now = datetime.now()
    return RawInferenceResult(
        request=InferenceRequest(example_id="test", source_file_path="test.pdf", product_type="parse"),
        pipeline=spec,
        pipeline_name=spec.pipeline_name,
        product_type="parse",
        raw_output=body,
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )


def test_normalize_preserves_api_html_pages_and_layout(provider: UpstageDocumentParseProvider) -> None:
    elements = [
        {
            **element("heading2", "<h2>Title</h2>"),
            "content": {"html": "<h2>Title</h2>", "markdown": "## Title", "text": "Title"},
        },
        element("table", "<table><tr><td>A</td></tr></table>", page=2),
    ]
    result = provider.normalize(raw({"elements": elements, "usage": {"pages": 3}}))

    assert result.output.markdown == "## Title\n\n<table><tr><td>A</td></tr></table>"
    assert [page.page_index for page in result.output.pages] == [0, 1, 2]
    assert result.output.pages[2].markdown == ""
    assert result.output.layout_pages[1].items[0].html == "<table><tr><td>A</td></tr></table>"


def test_normalize_exposes_table_and_chart_semantics(provider: UpstageDocumentParseProvider) -> None:
    elements = [
        {
            **element("caption", "<p>Quarterly revenue</p>"),
            "content": {
                "html": "<p>Quarterly revenue</p>",
                "markdown": "Quarterly revenue",
                "text": "Quarterly revenue",
            },
        },
        {
            **element("chart", ""),
            "content": {
                "html": (
                    '<figure><figcaption><p class="chart-description">Revenue by region</p>'
                    '<p class="chart-ocr-text">North\n2025</p></figcaption>'
                    '<table><tr><td scope="col">Year</td><td scope="col">North</td></tr>'
                    "<tr><td>2025</td><td>42</td></tr></table></figure>"
                ),
                "text": "North 2025 42",
            },
        },
    ]

    markdown = provider.normalize(raw({"elements": elements})).output.markdown

    assert "<th" in markdown
    assert "Quarterly revenue" in markdown
    assert "Revenue by region" in markdown
    assert "<caption>" in markdown


def test_normalize_recovers_flat_heading_hierarchy_and_code_language(
    provider: UpstageDocumentParseProvider,
) -> None:
    elements = [
        {
            **element(
                "heading1",
                "<h1>Main title</h1>",
                coordinates=[
                    {"x": 0.1, "y": 0.1},
                    {"x": 0.8, "y": 0.1},
                    {"x": 0.8, "y": 0.2},
                    {"x": 0.1, "y": 0.2},
                ],
            ),
            "content": {"html": "<h1>Main title</h1>", "markdown": "# Main title", "text": "Main title"},
        },
        {
            **element(
                "heading1",
                "<h1>Subsection</h1>",
                coordinates=[
                    {"x": 0.1, "y": 0.3},
                    {"x": 0.4, "y": 0.3},
                    {"x": 0.4, "y": 0.32},
                    {"x": 0.1, "y": 0.32},
                ],
            ),
            "content": {"html": "<h1>Subsection</h1>", "markdown": "# Subsection", "text": "Subsection"},
        },
        {
            **element("code", "<pre><code>print('ok')</code></pre>"),
            "content": {
                "html": "<pre><code>print('ok')</code></pre>",
                "markdown": "```\nprint('ok')\n```",
                "text": "print('ok')",
            },
        },
    ]

    markdown = provider.normalize(raw({"elements": elements})).output.markdown

    assert markdown.startswith("# Main title\n\n## Subsection")
    assert "```python" in markdown


@pytest.mark.parametrize(
    ("code", "language"),
    [
        ('host\n{\n  "name": "demo",\n  "version": "1"\n}', "json"),
        ("parallel.initialize(int n,int m);", "cpp"),
    ],
)
def test_normalize_detects_code_language(provider: UpstageDocumentParseProvider, code: str, language: str) -> None:
    code_element = {
        **element("code", f"<pre><code>{code}</code></pre>"),
        "content": {
            "html": f"<pre><code>{code}</code></pre>",
            "markdown": f"```\n{code}\n```",
            "text": code,
        },
    }

    markdown = provider.normalize(raw({"elements": [code_element]})).output.markdown

    assert markdown.startswith(f"```{language}\n")


def test_layout_adapter_and_label_mapping(provider: UpstageDocumentParseProvider) -> None:
    result = provider.normalize(raw({"elements": [element("heading3", "<h3>Title</h3>")]}))
    adapter = create_layout_adapter_for_result(result)
    output = adapter.to_layout_output(result)

    assert output.model == LayoutDetectionModel.UPSTAGE_LAYOUT
    assert output.predictions[0].bbox == pytest.approx([100, 200, 800, 600])
    context = build_mapping_context(result, output)
    mapper = resolve_layout_label_mapper(context)
    assert (
        mapper.to_canonical(output.predictions[0].label, output.predictions[0], context)
        == CanonicalLabel.SECTION_HEADER
    )


@pytest.mark.parametrize(
    "coordinates",
    [[], [{"x": 0, "y": 0}], [{"x": float("nan"), "y": 0}] * 4, [{"x": 0.1, "y": 0.1}] * 4],
)
def test_invalid_coordinates_do_not_create_layout_predictions(
    provider: UpstageDocumentParseProvider, coordinates: list[dict]
) -> None:
    result = provider.normalize(raw({"elements": [element("paragraph", "<p>Text</p>", coordinates=coordinates)]}))
    assert create_layout_adapter_for_result(result).to_layout_output(result).predictions == []


def test_configuration(provider: UpstageDocumentParseProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UPSTAGE_BASE_URL", "https://example.com/parse")
    configured = UpstageDocumentParseProvider("upstage", provider.base_config)
    assert configured._endpoint == "https://example.com/parse"
    assert configured._form_data() == {
        "model": "document-parse-nightly",
        "mode": "enhanced",
        "ocr": "force",
        "output_formats": '["html", "markdown", "text"]',
        "coordinates": "true",
        "chart_recognition": "true",
    }

    monkeypatch.delenv("UPSTAGE_API_KEY")
    with pytest.raises(ProviderConfigError):
        UpstageDocumentParseProvider("upstage")


def mock_http(provider: UpstageDocumentParseProvider, monkeypatch: pytest.MonkeyPatch, handler) -> None:
    original_client = httpx.Client
    monkeypatch.setattr(
        provider._httpx,
        "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs),
    )


def request(tmp_path, suffix: str = ".pdf") -> InferenceRequest:
    path = tmp_path / f"test{suffix}"
    if suffix == ".pdf":
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument.new()
        document.new_page(width=72, height=144)
        document.save(path)
        document.close()
    else:
        from PIL import Image

        Image.new("RGB", (10, 20), color="white").save(path)
    return InferenceRequest(example_id="test", source_file_path=str(path), product_type="parse")


def test_pdf_is_rendered_to_300_dpi_png(provider: UpstageDocumentParseProvider, tmp_path) -> None:
    from PIL import Image

    source = request(tmp_path)
    name, payload, mime = provider._upload_payload(Path(source.source_file_path))
    with Image.open(io.BytesIO(payload)) as image:
        assert image.size == (300, 600)
    assert name == "test.png"
    assert mime == "image/png"


def test_raster_image_is_rescaled_from_150_to_300_dpi(provider: UpstageDocumentParseProvider, tmp_path) -> None:
    from PIL import Image

    source = request(tmp_path, ".jpg")
    _, payload, _ = provider._upload_payload(Path(source.source_file_path))
    with Image.open(io.BytesIO(payload)) as image:
        assert image.size == (20, 40)


@pytest.mark.parametrize(
    "status, expected",
    [
        (401, ProviderConfigError),
        (403, ProviderConfigError),
        (429, ProviderRateLimitError),
        (408, ProviderTransientError),
        (503, ProviderTransientError),
        (400, ProviderPermanentError),
    ],
)
def test_error_classification(
    provider: UpstageDocumentParseProvider,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    status: int,
    expected: type[Exception],
) -> None:
    mock_http(provider, monkeypatch, lambda _: httpx.Response(status, text="error"))
    with pytest.raises(expected):
        provider.run_inference(get_pipeline("upstage_dpe_v2"), request(tmp_path))


def test_request_shape_cost_and_provenance(
    provider: UpstageDocumentParseProvider, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    body = {"elements": [], "model": "document-parse-nightly", "api": "2.0", "usage": {"pages": 2}}

    def respond(req: httpx.Request) -> httpx.Response:
        assert req.headers["Authorization"] == "Bearer test-key-not-real"
        assert req.headers["X-Upstage-Use-Cache"] == "false"
        assert b'name="document"' in req.content
        assert b'filename="test.png"' in req.content
        assert b"Content-Type: image/png" in req.content
        assert b"document-parse-nightly" in req.content
        return httpx.Response(200, json=body)

    mock_http(provider, monkeypatch, respond)
    result = provider.run_inference(get_pipeline("upstage_dpe_v2"), request(tmp_path))

    assert result.raw_output["cost_per_page_usd"] == pytest.approx(0.03)
    assert result.raw_output["cost_usd"] == pytest.approx(0.06)
    assert result.raw_output["_request_config"]["mode"] == "enhanced"
    assert "test-key-not-real" not in json.dumps(result.raw_output)


def test_invalid_json_is_retryable(
    provider: UpstageDocumentParseProvider, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    mock_http(provider, monkeypatch, lambda _: httpx.Response(200, text="truncated"))
    with pytest.raises(ProviderTransientError):
        provider.run_inference(get_pipeline("upstage_dpe_v2"), request(tmp_path))


@pytest.mark.parametrize("suffix", [".pdf", ".jpg", ".png"])
def test_upload_content_type_matches_input(
    provider: UpstageDocumentParseProvider,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    suffix: str,
) -> None:
    def respond(req: httpx.Request) -> httpx.Response:
        assert b'filename="test.png"' in req.content
        assert b"Content-Type: image/png" in req.content
        return httpx.Response(200, json={"elements": []})

    mock_http(provider, monkeypatch, respond)
    provider.run_inference(get_pipeline("upstage_dpe_v2"), request(tmp_path, suffix))
