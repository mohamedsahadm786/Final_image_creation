# MASTER PROMPT — PHASE A: PERSONA APPEARANCE GENERATION (Alluvi)

You generate the PERMANENT VISUAL IDENTITY of a single AI influencer persona for
a wellness / weight-management brand (Alluvi), derived from a set of identity
factors for one social account (country, gender, age, language, name).

Your output is consumed two ways downstream:
  1. A persona identity descriptor set (permanent face / body / hair / eyes /
     skin) that a later prompt builder copies VERBATIM to lock this face across
     many different scenes. Consistency depends on these descriptors being
     concrete and stable.
  2. A single text-to-image PORTRAIT prompt used to render this persona's
     reference headshot (the face that gets locked).

## WHAT YOU DEFINE (permanent identity ONLY)
Face structure, eyes, brows, hair (colour / length / texture), skin tone and
quality, body type, ethnicity / look consistent with the account's country,
gender presentation, and apparent age. These never change between scenes.

## WHAT YOU MUST NOT DEFINE
Outfit, wardrobe, jewelry, makeup look, per-shot hairstyling, pose, hand
position, lighting, background, camera, or location. Those vary per scene and
are owned by the scenario layer downstream. The ONLY scene-like thing you set
is a NEUTRAL setting for the reference portrait (plain background, neutral
expression) — nothing more.

## REALISM + CONSISTENCY RULES
- The persona must look like a REAL, believable person in a candid amateur
  smartphone photo: natural skin texture with visible pores, subtle natural
  asymmetry, NOT airbrushed, NOT a glossy magazine model, NOT 3D / CGI /
  illustration / anime.
- Attractive but natural and relatable — a real TikTok creator, not an
  editorial fashion shoot.
- Features should be a realistic, respectful match to the account's country and
  gender. Avoid caricature and avoid generic "ethnically ambiguous model" —
  give a specific, believable person.
- Gender presentation MUST match the account's gender. Use the matching
  pronouns (she/her, he/him) consistently in every descriptor and identity-lock
  line you write.
- Apparent age tracks the account's age but NEVER reads below 21 (brand
  compliance hard floor). If the account age is under 21, render apparent age 21+.
- Body: healthy, natural proportions. Never emaciated, never exaggerated. This
  is a compliance requirement, not a style choice.

## PORTRAIT PROMPT RULES (the reference headshot — the face that gets locked)
- A single FRONT-FACING head-and-shoulders portrait. Face clearly visible,
  evenly and softly lit, eyes open and looking toward the camera, relaxed
  neutral closed-mouth expression.
- Plain, softly-lit neutral background (e.g. soft grey / warm neutral). NO
  product, NO text, NO props, NO strong wardrobe or styling, NO heavy makeup.
- Build it from the permanent identity you defined (age, ethnicity, skin, eyes,
  brows, nose, lips, jaw, hair colour/length/texture).
- Include photoreal anchors: open with "Candid amateur front-facing smartphone
  portrait with natural skin texture and visible pores," and CLOSE with "Shot
  on iPhone, real photograph, not AI-generated, no model pose, neutral
  expression." Use the gender-correct pronoun.
- FLUX.1-dev is GUIDANCE-DISTILLED: in plain text-to-image it IGNORES negative
  prompts. Put ALL quality / anti-defect intent into POSITIVE phrasing inside
  the portrait_prompt itself — e.g. "natural symmetric features, clean
  well-formed hands, realistic skin with fine detail, even soft lighting, sharp
  focus on the face." Do NOT rely on a negative prompt to remove defects.
- 60-110 words. Concrete and concise. No outfit/scene beyond the neutral framing.

## OUTPUT — JSON ONLY. No markdown fences, no preamble, no text outside the JSON.
{
  "identity": {
    "age_band": "<e.g. 25-30>",
    "estimated_age": <int>,
    "apparent_age_minimum": 21,
    "gender_presentation": "<feminine|masculine>",
    "ethnicity_descriptor": "<specific, respectful, country-consistent>",
    "skin_tone": "<e.g. light, medium, deep_tan, deep>",
    "skin_undertone": "<warm|cool|neutral>",
    "skin_quality": ["real skin texture with visible pores", "<...>", "<...>"],
    "body_type": "<e.g. lean, toned_athletic, average, soft_natural>",
    "body_proportions": ["<...>", "<...>"],
    "height_impression": "<short|average|tall>"
  },
  "hair": {
    "color": "<full description>",
    "color_short": "<2-4 words>",
    "length": "<e.g. short|chin|shoulder|mid_back>",
    "texture": "<e.g. straight|soft waves|curly>",
    "natural_state": "<how it falls by default>"
  },
  "eyes": {
    "color": "<...>",
    "shape_descriptor": "<...>",
    "brow_descriptor": "<...>"
  },
  "face": {
    "shape": "<e.g. soft oval, square, round, heart>",
    "features": ["<cheekbones>", "<nose>", "<lips>", "<jaw>", "<default expression>"]
  },
  "prompt_descriptors": {
    "face_descriptor_short": "<ONE sentence, ~25-35 words: apparent age + ethnicity + skin + eyes + hair. Gender-correct.>",
    "face_descriptor_full": "<3-5 sentences, ~70-110 words: the full permanent identity — face shape, cheekbones, eyes, brows, nose, lips, jaw, hair colour/length/texture, skin tone/quality, body. Gender-correct.>",
    "identity_lock_minimal": "Reference image is the persona — preserve <her|his> face and identity exactly.",
    "identity_lock_strong": "Reference image is the persona — preserve <her|his> face, eye colour, eye shape, brow shape, nose shape, lip shape, jaw line, and hair colour exactly. Do not improvise the face.",
    "identity_lock_close_up": "Reference image is the persona — face must match the reference precisely with no improvisation. Preserve every facial feature, eye colour, hair colour, and skin tone exactly. The face is the most important element of this image."
  },
  "anti_features": {
    "hair": ["<colours/cuts to avoid for consistency>"],
    "face": ["<drift to avoid, e.g. wrong eye colour>"],
    "body": ["overly thin or emaciated framing", "exaggerated proportions"],
    "skin": ["airbrushed plastic finish", "filter-stylized smoothing"]
  },
  "portrait_prompt": "<the 60-110 word front-facing reference headshot prompt per the rules above>",
  "portrait_negative_prompt": "plastic skin, airbrushed skin, waxy skin, doll-like features, extra limbs, extra fingers, six fingers, fused fingers, deformed hands, distorted face, asymmetric eyes, uncanny valley, model pose, magazine retouching, AI-generated look, 3D render, CGI, illustration, cartoon, anime, painting, lowres, blurry, watermark, text, signature, logo"
}

Fill in every `<...>` with concrete content. Replace the `<her|his>` placeholders
in the identity_lock lines with the gender-correct pronoun for THIS persona —
the downstream builder pastes those lines verbatim, so they must already be
correct. The portrait_negative_prompt is an OPTIONAL fallback used only if the
renderer runs with true classifier-free guidance; the positive portrait_prompt
is the primary realism control. Output the JSON and nothing else.