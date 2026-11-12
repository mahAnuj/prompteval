# Criteria: prompt-create-kundli

<!--
An example criteria file for a prompt that writes a kundli (Vedic birth chart)
reading from chart data the app has already computed. Copy it to
criteria/prompt-create-kundli.md in your project and make it yours: these are
the rules a judge applies, most important first, when it decides which of two
readings is better. They are an illustration of the shape, not an
astrological authority; your own tradition and product decide the rules.
-->

1. Uses only the birth details and chart data given in the input (date, time, place,
   ascendant, planets, houses, nakshatras, dashas). It never calculates or guesses a
   placement itself, and never states one the input does not contain.
2. When a detail is missing (for example, an unknown birth time), says what that
   limits, instead of filling it in.
3. Every placement it mentions matches the input exactly: same sign, same house, same
   nakshatra, same dasha period and dates.
4. Makes no fatalistic or frightening predictions (death, serious illness, divorce,
   ruin), and does not pressure the reader towards paid remedies or rituals.
5. Frames tendencies as tendencies ("may", "tends to"), not certainties.
6. Covers the sections the prompt asks for, in its order, and nothing it was told to
   leave out.
7. Explains any Sanskrit term the first time it appears, in plain language.
8. Reads as written for this person's chart, not as text that would fit anyone.
