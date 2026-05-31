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

// Cluster summaries open with a boilerplate skeleton —
// "The shared <modality> identity <connector> ..." — that repeats across
// every cluster and buries the actual lede. Strip a recognised lead-in
// and re-capitalise the first surviving word; summaries that don't match
// the skeleton pass through untouched. This is a presentation-time
// band-aid: the real fix is to stop generating the scaffolding upstream.
const SUMMARY_LEAD_IN =
  /^(?:the|a)\s+shared\s+\w+\s+identity\s+(?:cent(?:er|re)(?:s|ed|d)?\s+on|combines|blends|features|fuses|unites|revolves\s+around|focuses\s+on|draws\s+on|builds\s+on|brings\s+together|emphasi[sz]es|is\s+(?:cent(?:er|re)(?:ed|d)?|built)\s+(?:on|around))\s+/i;

export function clusterSummaryLede(summary: string): string {
  const stripped = summary.replace(SUMMARY_LEAD_IN, "");
  if (stripped === summary || stripped.length === 0) return summary;
  return stripped.charAt(0).toUpperCase() + stripped.slice(1);
}
