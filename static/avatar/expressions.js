// static/avatar/expressions.js — Mapping tratti → morph targets (spec Phase 1.6)
// Importato da face.js come sorgente di verità per le espressioni.

export const EXPRESSION_MAP = {
  curiosity: { brow_raise: 0.6, eye_wide: 0.4, head_tilt: 0.2 },
  fear:      { brow_inner: 0.7, eye_wide: 0.8, mouth_tense: 0.3 },
  warmth:    { smile: 0.5, eye_soft: 0.4, head_forward: 0.1 },
  cynicism:  { brow_lower: 0.4, eye_squint: 0.3, head_back: 0.2 },
  trust:     { eye_contact: 1.0, smile: 0.2 },
};
