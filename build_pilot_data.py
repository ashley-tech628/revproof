"""Build a small, source-linked real revision pilot from manually selected edits.

Rows are normalized document excerpts, not claims that the entire API behavior
was independently tested. The fixed split holds out Node release transitions
and Flask as an external document family.
"""
import json
import re
from pathlib import Path

from revproof_data import read_records


# id, split, field, question, answer before/after, old/new cue, preserved facts, edit type
SPECS = [
    ("node_http_v16.20.2_v17.9.1_0190", "train", "Content-Length validation",
     "Does Node.js check that Content-Length equals the transmitted body length?",
     "yes", "no", "will check whether", "does not check whether",
     ["Use [`Buffer.byteLength()`][] to determine the length of the body in bytes."], "conditional"),
    ("node_http_v16.20.2_v17.9.1_0191", "train", "invalid HTTP header error",
     "What error type is documented for an invalid HTTP header name or value?",
     "Error", "TypeError", "Error", "TypeError",
     ["Attempting to set a header field name or value that contains invalid characters"], "scalar"),
    ("node_http_v17.9.1_v18.20.8_0092", "train", "clientError socket closure",
     "By when must a clientError listener close or destroy the socket?",
     "responsibility stated, no deadline", "before the listener ends",
     "closing/destroying the underlying socket", "**must be closed or destroyed**",
     ["If a client connection emits an `'error'` event, it will be forwarded here."], "cross_sentence"),
    ("node_http_v17.9.1_v18.20.8_0094", "train", "clientError immediate destruction",
     "Under what sent-response condition is the clientError socket immediately destroyed?",
     "socket has already written data", "response headers have been sent",
     "has already written data", "headers of the current attached",
     ["Default behavior is to try close the socket"], "conditional"),
    ("node_fs_v17.9.1_v18.20.8_0101", "train", "NTFS junction target type",
     "What may NTFS junction points target?", "not specified", "directories only",
     "Windows junction points require the destination path to be absolute.",
     "Junction points on NTFS volumes can only point to directories.",
     ["Windows junction points require the destination path to be absolute."], "cross_sentence"),
    ("node_stream_v17.9.1_v18.20.8_0179", "train", "pipeline listener cleanup",
     "What happens to dangling listeners if the last pipeline stream is readable?",
     "left dangling after callback", "removed so the last stream can be consumed later",
     "leaves dangling event listeners", "dangling event listeners will be removed",
     ["In the case of reuse of streams after failure, this can cause event listener leaks and swallowed errors."],
     "conditional"),
    ("node_url_v17.9.1_v18.20.8_0060", "train", "URLSearchParams.delete condition",
     "When does delete remove every pair with the given name?",
     "name-only behavior stated", "when value is not provided",
     "Remove all name-value pairs whose name is `name`.",
     "If `value` is not provided, removes all name-value pairs whose name is `name`.",
     [], "conditional"),
    ("node_url_v17.9.1_v18.20.8_0066", "train", "URLSearchParams.has condition",
     "When does has check for at least one pair with the given name?",
     "name-only behavior stated", "when value is not provided",
     "Returns `true` if there is at least one name-value pair whose name is `name`.",
     "If `value` is not provided, returns `true` if there is at least one name-value pair whose name is `name`.",
     [], "conditional"),
    ("node_buffer_v17.9.1_v18.20.8_0034", "train", "Buffer.allocUnsafe range error",
     "Which error code is documented when allocUnsafe size is out of range?",
     "ERR_INVALID_ARG_VALUE", "ERR_OUT_OF_RANGE", "ERR_INVALID_ARG_VALUE", "ERR_OUT_OF_RANGE",
     ["Allocates a new `Buffer` of `size` bytes."], "scalar"),
    ("node_errors_v19.9.0_v20.19.0_0403", "train", "HTTP header overflow threshold",
     "What threshold triggers HPE_HEADER_OVERFLOW?", "8 KiB", "maxHeaderSize",
     "8 KiB", "`maxHeaderSize`",
     ["HTTP parsing will abort without a request or response object being created"], "scalar"),
    ("node_crypto_v19.9.0_v20.19.0_0302", "train", "custom OpenSSL engine status",
     "What is the status of custom engines starting with OpenSSL 3?",
     "deprecation not stated", "deprecated",
     "Load and set the `engine` for some or all OpenSSL functions (selected by flags).",
     "Support for custom engines in OpenSSL is deprecated from OpenSSL 3.",
     ["Load and set the `engine` for some or all OpenSSL functions (selected by flags)."],
     "cross_sentence"),
    ("node_stream_v19.9.0_v20.19.0_0016", "train", "string-stream highWaterMark unit",
     "What unit does highWaterMark use for strings that are not decoded?",
     "not stated", "UTF-16 code units",
     "For streams operating in object mode, the `highWaterMark` specifies a total number of objects.",
     "For streams operating on (but not decoding) strings, the `highWaterMark` specifies a total number of UTF-16 code units.",
     ["For streams operating in object mode, the `highWaterMark` specifies a total number of objects."],
     "cross_sentence"),
    ("node_stream_v19.9.0_v20.19.0_0204", "train", "default byte-mode highWaterMark",
     "What is the documented default highWaterMark for non-object streams?",
     "16384 bytes (16 KiB)", "65536 bytes (64 KiB)",
     "`16384` (16 KiB)", "`65536` (64 KiB)",
     ["or `16` for `objectMode`"], "scalar"),
    ("node_stream_v19.9.0_v20.19.0_0248", "train", "readable.push accepted chunks",
     "Which non-string chunk types does readable.push document?",
     "Buffer and Uint8Array", "Buffer, TypedArray, and DataView",
     "`Buffer`, `Uint8Array`, or `string`", "{Buffer}, {TypedArray}, {DataView} or {string}",
     ["Passing `chunk` as `null` signals the end of the stream"], "scalar"),
    ("node_process_v19.9.0_v20.19.0_0076", "train", "constrainedMemory unconstrained result",
     "What does constrainedMemory return if the constraint is absent or unknown?",
     "undefined", "0", "`undefined`", "`0`",
     ["based on limits imposed by the OS"], "scalar"),
    ("node_http_v21.7.3_v22.14.0_0182", "dev", "nonstandard response body condition",
     "When does writing a body for HEAD, 204, or 304 synchronously throw?",
     "unconditionally according to the excerpt", "when rejectNonStandardBodyWrites is true",
     "Writing to the body is not allowed",
     "If `rejectNonStandardBodyWrites` is set to true in `createServer` then writing to the body is not allowed",
     ["a synchronous `Error` with the code `ERR_HTTP_BODY_NOT_ALLOWED` is thrown."], "conditional"),
    ("node_buffer_v21.7.3_v22.14.0_0051", "dev", "Buffer.concat short-list padding",
     "What fills unused space if Buffer.concat totalLength exceeds the listed buffers?",
     "not stated", "zeros",
     "the result is truncated to `totalLength`.",
     "the remaining space is filled with zeros.",
     ["the result is truncated to `totalLength`."], "cross_sentence"),
    ("node_buffer_v21.7.3_v22.14.0_0217", "dev", "64-bit Buffer maximum length",
     "What is the documented MAX_LENGTH on 64-bit architectures?",
     "2^32 (about 4 GiB)", "2^53 - 1 (about 8 PiB)",
     "2<sup>32</sup> (about 4 GiB)", "2<sup>53</sup> - 1 (about 8 PiB)",
     ["On 64-bit architectures"], "scalar"),
    ("node_net_v21.7.3_v22.14.0_0133", "dev", "autoSelectFamilyAttemptTimeout default",
     "What can determine the initial default autoSelectFamilyAttemptTimeout?",
     "250", "250 or the command-line option",
     "The initial default value is `250`.",
     "`--network-family-autoselection-attempt-timeout`",
     ["The initial default value is `250`"], "conditional"),
]

# A question about an unchanged fact uses the same revision group and target span.
KEEP = {
    "node_stream_v19.9.0_v20.19.0_0204":
        ("What is the documented default for objectMode?", "16"),
    "node_stream_v19.9.0_v20.19.0_0016":
        ("What does highWaterMark count in object mode?", "objects"),
    "node_buffer_v21.7.3_v22.14.0_0051":
        ("What happens if the listed buffers exceed totalLength?", "truncated to totalLength"),
}

# These are additions or qualifications: the old cue remains true in the new
# paragraph. Removing it would itself be an erroneous edit.
ALLOW_OLD_VALUE = {
    "node_http_v17.9.1_v18.20.8_0092",
    "node_fs_v17.9.1_v18.20.8_0101",
    "node_stream_v17.9.1_v18.20.8_0179",
    "node_crypto_v19.9.0_v20.19.0_0302",
    "node_stream_v19.9.0_v20.19.0_0016",
    "node_buffer_v21.7.3_v22.14.0_0051",
}


def load_sources(path):
    return {r["candidate_id"]: r for r in
            (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)}


def node_rows(root):
    sources = load_sources(root / "data" / "node_selected_sources.jsonl")
    rows = []
    for (candidate_id, split, field, question, before, after, old_value, new_value,
         protected, edit_type) in SPECS:
        source = sources[candidate_id]
        doc = source["document"]
        heading = source["heading_old"]
        base = {"group": candidate_id, "split": split, "source_type": "real",
                "source_url": source["source_url_new"],
                "source_url_before": source["source_url_old"],
                "source_sha256_before": source["source_sha256_old"],
                "source_sha256_after": source["source_sha256_new"],
                "source_path": f"doc/api/{doc}.md",
                "prefix": f"# Node.js {doc}\n\n## {heading}\n\n",
                "old_span": source["old_span"], "new_span": source["new_span"],
                "suffix": "", "field": field, "old_value": old_value,
                "new_value": new_value,
                "version_before": source["version_before"],
                "version_after": source["version_after"],
                "scope": f"Node.js {doc} API: {heading}",
                "protected_facts": protected, "edit_type": edit_type,
                "allow_old_value_in_new_span": candidate_id in ALLOW_OLD_VALUE,
                "audit_note": "Selected from official tagged Node.js API Markdown; line wraps normalized. Runtime behavior was not independently tested."}
        rows.append({**base, "id": candidate_id + "_change", "question": question,
                     "answer_before": before, "answer_after": after})
        if candidate_id in KEEP:
            keep_question, keep_answer = KEEP[candidate_id]
            rows.append({**base, "id": candidate_id + "_keep", "question": keep_question,
                         "answer_before": keep_answer, "answer_after": keep_answer})
    return rows


def flask_rows(root):
    sources = load_sources(root / "data" / "flask_selected_sources.jsonl")
    prior = [json.loads(line) for line in (root / "data" / "flask_pilot_audited.jsonl")
             .read_text(encoding="utf-8").splitlines() if line]
    for row in prior:
        row["split"] = "test"
    annotations = [
        ("c12a5d87_ab814966_0000", "SERVER_NAME subdomain matching",
         "When must SERVER_NAME be set for subdomain matching?",
         "required for subdomain route matching support", "when subdomain_matching is enabled",
         "Required for subdomain route matching support.",
         "Must be set if ``subdomain_matching`` is enabled",
         ["Inform the application what host and port it is bound to."], "conditional"),
        ("c12a5d87_ab814966_0002", "MAX_CONTENT_LENGTH per-request override",
         "Can MAX_CONTENT_LENGTH be customized for a specific request?",
         "only app config described", "yes, through Request.max_content_length",
         "Don't read more than this many bytes from the incoming request data.",
         "It can be set on a specific",
         [], "cross_sentence"),
    ]
    for (candidate_id, field, question, before, after, old_value, new_value,
         protected, edit_type) in annotations:
        source = sources[candidate_id]
        old_span = re.sub(r"\s+", " ", " ".join(source["old_lines"])).strip()
        new_span = re.sub(r"\s+", " ", " ".join(source["new_lines"])).strip()
        prior.append({"id": "flask_" + candidate_id + "_change",
                      "group": "flask_" + candidate_id, "split": "test",
                      "source_type": "real",
                      "source_url": "https://github.com/pallets/flask/compare/3.0.3...3.1.0",
                      "source_path": source["path"], "question": question,
                      "answer_before": before, "answer_after": after,
                      "prefix": f"Flask configuration: {field}\n\n",
                      "old_span": old_span, "new_span": new_span, "suffix": "",
                      "field": field, "old_value": old_value,
                      "new_value": new_value,
                      "version_before": "Flask 3.0.3", "version_after": "Flask 3.1.0",
                      "scope": "Flask configuration documentation",
                      "protected_facts": protected, "edit_type": edit_type,
                      "audit_note": "Manually selected versioned documentation hunk; 3.1.0 release notes corroborate the change."})
    return prior


def main():
    root = Path(__file__).parent
    output = root / "data" / "real.jsonl"
    rows = node_rows(root) + flask_rows(root)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                      encoding="utf-8")
    checked = read_records(output)
    groups = {s: len({r["group"] for r in checked if r["split"] == s})
              for s in ("train", "dev", "test")}
    print(f"Wrote {len(checked)} real pilot rows to {output}; revision groups: {groups}")


if __name__ == "__main__":
    main()
