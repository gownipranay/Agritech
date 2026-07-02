// Static fallback so the UI still renders crop choices if the backend is
// asleep (Render free tier cold-starts). The backend /crops response, when
// reachable, overrides this.
export const CROP_EMOJI = {
  tomato: "🍅",
  potato: "🥔",
  bell_pepper: "🫑",
  chilli: "🌶️",
  corn: "🌽",
  paddy: "🌾",
  toor_dal: "🫘",
  groundnut: "🥜",
  cotton: "☁️",
};

export const FALLBACK_CROPS = [
  { id: "tomato", name: "Tomato", vision_support: "vision_supported" },
  { id: "potato", name: "Potato", vision_support: "vision_supported" },
  { id: "bell_pepper", name: "Bell Pepper", vision_support: "vision_supported" },
  { id: "chilli", name: "Chilli", vision_support: "vision_supported" },
  { id: "corn", name: "Corn / Maize", vision_support: "vision_supported" },
  { id: "paddy", name: "Paddy / Rice", vision_support: "vision_supported" },
  { id: "toor_dal", name: "Toor Dal", vision_support: "advisory_only" },
  { id: "groundnut", name: "Groundnut", vision_support: "advisory_only" },
  { id: "cotton", name: "Cotton", vision_support: "vision_supported" },
];

export const SUPPORT_LABEL = {
  vision_supported: { text: "Photo detection", cls: "supported" },
  vision_proxy: { text: "Photo (via pepper)", cls: "proxy" },
  advisory_only: { text: "Advisory only", cls: "advisory" },
};
