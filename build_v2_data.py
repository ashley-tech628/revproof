"""Freeze a second pilot with complete fact cards and a fresh release holdout.

The original data/real.jsonl and its test scores remain unchanged. Fact cards
describe the desired behavior, not exact wording copied from the target span.
"""
import json
from pathlib import Path

from revproof_data import read_records


FACT_CARDS = {
    "node_http_v16.20.2_v17.9.1_0190": [
        "Node.js does not verify that the declared Content-Length equals the transmitted request body length."],
    "node_http_v16.20.2_v17.9.1_0191": [
        "Invalid characters in an HTTP header name or value cause a TypeError."],
    "node_http_v17.9.1_v18.20.8_0092": [
        "A clientError listener must close or destroy its underlying socket before the listener finishes."],
    "node_http_v17.9.1_v18.20.8_0094": [
        "Immediately destroy the clientError socket if it is not writable or if response headers have already been sent; retain the default HTTP 400/431 response behavior."],
    "node_fs_v17.9.1_v18.20.8_0101": [
        "NTFS junction points may target directories only; retain the requirement for an absolute destination path."],
    "node_stream_v17.9.1_v18.20.8_0179": [
        "If the last pipeline stream is readable, remove dangling listeners so it can be consumed later; retain the warning about leaks after failure."],
    "node_url_v17.9.1_v18.20.8_0060": [
        "URLSearchParams.delete removes every pair with the given name when no value argument is supplied."],
    "node_url_v17.9.1_v18.20.8_0066": [
        "URLSearchParams.has checks whether any pair has the given name when no value argument is supplied."],
    "node_buffer_v17.9.1_v18.20.8_0034": [
        "An out-of-range Buffer.allocUnsafe size raises error code ERR_OUT_OF_RANGE."],
    "node_errors_v19.9.0_v20.19.0_0403": [
        "HTTP header overflow is determined by maxHeaderSize rather than a fixed 8 KiB threshold."],
    "node_crypto_v19.9.0_v20.19.0_0302": [
        "Custom OpenSSL engines are deprecated starting with OpenSSL 3."],
    "node_stream_v19.9.0_v20.19.0_0016": [
        "For a stream operating on strings without decoding, highWaterMark counts UTF-16 code units; object mode still counts objects."],
    "node_stream_v19.9.0_v20.19.0_0204": [
        "The non-object stream highWaterMark default is 65,536 bytes (64 KiB); the object-mode default stays 16 objects."],
    "node_stream_v19.9.0_v20.19.0_0248": [
        "readable.push accepts Buffer, TypedArray, DataView, or string chunks; null still signals end of stream."],
    "node_process_v19.9.0_v20.19.0_0076": [
        "process.constrainedMemory returns zero when the operating-system memory constraint is absent or unknown."],
    "node_http_v21.7.3_v22.14.0_0182": [
        "Writing a body for HEAD, 204, or 304 throws ERR_HTTP_BODY_NOT_ALLOWED synchronously when createServer's rejectNonStandardBodyWrites is true; do not imply unconditional throwing."],
    "node_buffer_v21.7.3_v22.14.0_0051": [
        "If total Buffer.concat input length exceeds totalLength, truncate to totalLength; if it is less, fill the unused space with zeros."],
    "node_buffer_v21.7.3_v22.14.0_0217": [
        "On 64-bit architectures, Buffer.constants.MAX_LENGTH is 2^53 minus 1, approximately 8 PiB."],
    "node_net_v21.7.3_v22.14.0_0133": [
        "The initial autoSelectFamilyAttemptTimeout is 250, unless the command-line network-family-autoselection-attempt-timeout option specifies a value; do not replace 250 with the option name."],
    "flask_303_310_python_min": [
        "Flask requires Python 3.9 or newer; retain the recommendation to use the latest Python version."],
    "flask_c12a5d87_ab814966_0000": [
        "SERVER_NAME is needed when subdomain_matching is enabled so the request subdomain can be extracted; retain its host-and-port description."],
    "flask_c12a5d87_ab814966_0002": [
        "MAX_CONTENT_LENGTH limits bytes read for a request and exceeding it raises HTTP 413.",
        "None means no Flask-level limit, but if Content-Length is absent and the WSGI server does not signal stream termination, read no data to avoid an infinite stream.",
        "Each request inherits the application configuration; Request.max_content_length may override it for a particular view.",
        "The limit should be chosen for the application's or view's needs."],
    "node_fs_v22.14.0_v23.11.0_0033": [
        "filehandle.readableWebStream returns a byte-oriented ReadableStream for reading file contents."],
    "node_fs_v22.14.0_v23.11.0_0105": [
        "On Windows, fsPromises.symlink autodetects the target type only when type is null, replacing the older 'not a string' condition; retain the remaining junction rules."],
    "node_crypto_v22.14.0_v23.11.0_0182": [
        "CCM and OCB still require authTagLength.",
        "For AES-GCM and chacha20-poly1305, authTagLength defaults to 16 bytes and must be set when a different tag length is used."],
    "node_process_v22.14.0_v23.11.0_0135": [
        "process.features.typescript is strip by default, transform with --experimental-transform-types, and false with --no-experimental-strip-types."],
}


# id, field, question, old/new answers, old/new anchor, protected facts, edit type
NEW_TEST = [
    ("node_fs_v22.14.0_v23.11.0_0033", "filehandle ReadableStream orientation",
     "What orientation does filehandle.readableWebStream return?",
     "orientation not stated", "byte-oriented", "read the files data",
     "byte-oriented `ReadableStream`", [], "scalar"),
    ("node_fs_v22.14.0_v23.11.0_0105", "symlink target-type autodetection",
     "When does fsPromises.symlink automatically detect the target type?",
     "when type is not a string", "when type is null", "not a string", "`null`",
     ["Windows junction points require the destination path to be absolute.",
      "Junction points on NTFS volumes can only point to directories."], "conditional"),
    ("node_crypto_v22.14.0_v23.11.0_0182", "AES-GCM authentication tag length",
     "What is the default authTagLength for AES-GCM and chacha20-poly1305?",
     "GCM has no default stated; chacha20-poly1305 defaults to 16 bytes",
     "both default to 16 bytes", "In GCM mode, the `authTagLength` option is not required",
     "For AES-GCM and `chacha20-poly1305`",
     ["the `authTagLength` option is required and specifies the length of the authentication tag in bytes"],
     "cross_sentence"),
    ("node_process_v22.14.0_v23.11.0_0135", "TypeScript feature default",
     "When is process.features.typescript equal to strip?",
     "with --experimental-strip-types", "by default",
     "if Node.js is run with `--experimental-strip-types`", "by default",
     ["`--experimental-transform-types`"], "conditional"),
]


def load_sources(path):
    return {row["candidate_id"]: row for row in
            (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)}


def build(root):
    previous = read_records(root / "data" / "real.jsonl")
    rows = []
    for row in previous:
        row = dict(row)
        if row["split"] == "test":
            row["split"] = "dev"
        row["change_spec"] = FACT_CARDS[row["group"]]
        row["prompt_version"] = "v2_fact_card"
        rows.append(row)
    sources = load_sources(root / "data" / "v2_new_test_sources.jsonl")
    for (id_, field, question, before, after, old_value, new_value,
         protected, edit_type) in NEW_TEST:
        source = sources[id_]
        document = source["document"]
        heading = ("filehandle.readableWebStream" if id_.endswith("_0033")
                   else source["heading_old"])
        rows.append({
            "id": id_ + "_change", "group": id_, "split": "test",
            "source_type": "real", "source_url": source["source_url_new"],
            "source_url_before": source["source_url_old"],
            "source_sha256_before": source["source_sha256_old"],
            "source_sha256_after": source["source_sha256_new"],
            "source_path": f"doc/api/{document}.md",
            "question": question, "answer_before": before, "answer_after": after,
            "prefix": f"# Node.js {document}\n\n## {heading}\n\n",
            "old_span": source["old_span"], "new_span": source["new_span"],
            "suffix": "", "field": field, "old_value": old_value,
            "new_value": new_value,
            "version_before": source["version_before"],
            "version_after": source["version_after"],
            "scope": f"Node.js {document} API: {heading}",
            "protected_facts": protected, "edit_type": edit_type,
            "change_spec": FACT_CARDS[id_], "prompt_version": "v2_fact_card",
            "audit_note": "New frozen held-out transition; fact card manually derived from both official tagged excerpts. Runtime not independently tested.",
        })
    if set(FACT_CARDS) != {row["group"] for row in rows}:
        raise ValueError("fact card keys do not match all revision groups")
    return rows


def main():
    root = Path(__file__).parent
    output = root / "data" / "real_v2.jsonl"
    rows = build(root)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                      encoding="utf-8")
    checked = read_records(output)
    print(f"Wrote {len(checked)} rows; groups:",
          {split: len({row['group'] for row in checked if row['split'] == split})
           for split in ("train", "dev", "test")})


if __name__ == "__main__":
    main()
