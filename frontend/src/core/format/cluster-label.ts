const KNOWN_WARNINGS: Record<string, string> = {
  tag_count_out_of_range: "tag count out of range",
  tag_length_out_of_range: "tag length out of range",
  duplicate_tag_within_kind: "duplicate tag within kind",
  ungrounded_tag_reference: "ungrounded tag reference",
  invalid_confidence: "invalid confidence",
  sentence_length_out_of_range: "sentence length out of range",
  invalid_tool_tags: "invalid tool tags",
};

export function clusterWarningLabel(code: string): string {
  return KNOWN_WARNINGS[code] ?? code;
}
