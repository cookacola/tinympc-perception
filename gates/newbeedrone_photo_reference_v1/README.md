# NewBeeDrone Micro Race Gate photo texture v1

`newbeedrone_gate_front_uv_v1.png` is a front-facing 1024x1024 UV texture
cropped from the user-approved product reference. It intentionally preserves:

- the diagonal orange/black fabric seams across the rounded corners;
- the four orange honeycomb side panels;
- the two side bee marks;
- upright top branding and inverted bottom branding.

The Isaac Sim generator maps this texture onto separate front and back rounded
ring faces. A black edge mesh supplies 25 mm of physical depth, so the central
opening and rounded outer silhouette are geometry rather than transparency.

Approved source crop in the original 554x554 reference: `(x=107, y=97,
width=340, height=341)`. The UV texture SHA-256 is
`dda97a6d3322bc077f69f3453ffa6a2a86ff32ddf3d9e5cedd1b9ac9acdac592`.
