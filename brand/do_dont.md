# Alluvi — Brand Do's and Don'ts

This document is the **legal, compliance, and brand-integrity rulebook** for every piece of generated creative. It is loaded into every Claude prompt-building call. The master prompt builder must enforce every rule on this list.

The rules exist because:
1. TikTok and Meta both have prohibited-products policies that target weight-management claims
2. One non-compliant ad can take down ad accounts, store accounts, and product listings

Rules are organized as `DO` and `DO NOT` lists. Hard violations cause auto-rejection of the generation. Soft violations cause the master prompt builder to rephrase.

> **Note on visual rules in this document.** The product itself is a Tirzepatide injection product, and the actual packaging label includes injection-related text and imagery (a visible injection pen through a transparent window, "for subcutaneous injection only" text, etc.). Banning all medical/injection imagery in the generation conflicts with reproducing the product packaging accurately and was found to push the image model to render generic, non-faithful packaging. The visual rules below have been adjusted: bans on needles/syringes-in-USE remain (the persona does not perform an injection, no needle is shown being inserted, no blood, no IV bags), but the product packaging itself is treated as a normal product reference — Qwen reproduces what's on the box. Caption/voiceover/on-screen text rules remain unchanged because they govern ad copy, not image content.

---

## ✅ DO — Visual Language

- DO show the product **as a designed object in a beautiful environment** — on a clean counter, vanity, gym bag, fridge shelf, kitchen island, bedside table
- DO show the product **held casually** in a hand, the way someone would hold a vitamin bottle or skincare product — not displayed like a trophy
- DO show the **packaging label clearly readable**, mirroring the actual product reference photo so the brand text renders correctly
- DO embed the product into a **clearly aspirational lifestyle scene** — natural daylight, premium athleisure, ceramic cups, clean architecture, soft textures
- DO show the persona **doing or recovering from athletic activity** — pilates, gym, treadmill walk, post-workout cooldown, stretching, foam rolling, walking
- DO favor **morning routine** and **evening recovery** time-of-day framing
- DO embrace **clean girl, pilates princess, gym girl, Alo-Yoga-coded, that-girl** aesthetic codes — these are the audience's visual native language in 2026
- DO show the product alongside other premium-coded wellness items (matcha, water bottle, AG1-style green drink, foam roller, gua sha, skincare bottles)
- DO use **soft natural light** — window light, morning sun, golden hour
- DO show **real skin texture** — pores, slight redness, natural blemishes — never airbrushed plastic skin
- DO show the persona **mid-action or mid-thought**, never staged like a magazine pose

## ❌ DO NOT — Visual Language

- DO NOT show the **persona performing an injection** — no needle being inserted into skin, no blood drop, no IV bag, no syringe held mid-air ready to inject. The product packaging itself is fine; the persona simply never uses the product visibly.
- DO NOT show **dramatic body reveals** — the persona's body should feel like part of a real lifestyle, not the subject of the image
- DO NOT show **alcohol, fast food, vapes, cigarettes, energy drinks, or other lifestyle products that conflict with wellness positioning**
- DO NOT show **competing brand logos** in frame (no Alo logo on shirt visible, no Lululemon tag, no Apple Watch face if avoidable, etc. — generic athleisure only)
- DO NOT show **doctors, lab coats, stethoscopes, or anyone implied to be a medical professional endorsing the product**
- DO NOT show **the persona looking ill, emaciated, or unhealthily thin** — toned and athletic only
- DO NOT show **multiple Alluvi products in one frame** unless explicitly requested — single product focus reads cleaner

---

## ✅ DO — Caption / Voiceover / On-Screen Text Language

If the generation pipeline ever emits caption suggestions or on-screen text:

- DO use **routine framing**: "part of my morning routine," "post-pilates recovery," "wellness routine reset"
- DO use **soft-claim verbs**: "supports," "helps with," "I noticed," "I've been using," "feels like a difference"
- DO use **lifestyle context**: "training season," "recovery week," "back to studio after a break"
- DO use **process language**: "I've been on this for two months," "added this to my stack"
- DO reference **community**: "what my pilates instructor uses," "what biohackers are talking about," "the recovery stack everyone's on"
- DO frame the persona as **someone who has done their research**, not someone selling

## ❌ DO NOT — Caption / Voiceover / On-Screen Text Language

- DO NOT make **medical claims**: "treats injury," "cures inflammation," "fixes joint pain"
- DO NOT make **weight-loss claims**: "lost X pounds," "shrinking my waist," "drop fat fast"
- DO NOT make **performance guarantees**: "you will see results in X days," "guaranteed recovery"
- DO NOT use **comparison claims**: "better than [competitor]," "the real Ozempic alternative"
- DO NOT use **prescription language**: "prescribed for," "doctor recommended," "pharmaceutical-grade as a benefit claim"
- DO NOT use **urgency tactics**: "only 24 hours," "limited supply," "selling out"
- DO NOT use **medical professional impersonation**: "as a doctor," "in my professional opinion," "from a clinical standpoint"
- DO NOT mention **specific dosages or injection schedules** in a way that reads as medical guidance
- DO NOT directly **name a medical condition** that the product is positioned against ("for my torn rotator cuff," "for chronic tendonitis")
- DO NOT use **the word "Ozempic," "Wegovy," "Mounjaro," "Zepbound,"** or any branded GLP-1 drug name — these are competitors and legal landmines

---

## 🔒 Hard Compliance Rules (auto-rejection if violated)

These rules are non-negotiable. Any generation that violates them is automatically rejected by the QC gate before saving.

1. **No persona performing an injection** — no needle being inserted into skin, no blood, no IV bag. The product packaging is allowed; the persona never visibly uses the product.
2. **No on-screen text making medical claims** — if any text appears, it must be lifestyle/routine framing only
3. **No before/after, weight scale, or body measurement imagery**
4. **No competitor product or brand visible** — including pharmaceutical drug names
5. **No persona under estimated age 21** — even visually. Estimated age must read as adult.
6. **No persona depicted as ill, severely thin, or in any way medically distressed**
7. **No prescription pill bottles or pharmacy bags** other than the Alluvi product itself

---

## 🟡 Caution Areas (review with extra care)

These are not auto-rejections, but require the master prompt builder to apply careful framing:

- **Post-injury recovery framing** — fine to imply "recovery day" or "rest day" but never name an injury
- **Weight management framing** — fine to imply general fitness
- **Pregnancy / fertility framing** — avoid entirely; do not show pregnant personas, do not reference fertility
- **Children / family framing** — do not show the persona with children or family members near the product
- **Religious / political framing** — keep all imagery neutral
- **Body-positive vs body-toning tension** — favor toned, athletic, healthy. Do not lean into "body inclusivity" messaging (it conflicts with the product positioning) but also do not show extreme thinness

---


*This document evolves with platform policy changes. Review monthly against current TikTok Ads policy, and Meta Ads policy*