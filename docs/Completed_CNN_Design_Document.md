The goal of this CNN is to be the perception frontend of a control and state estimation pipeline, feeding direct half-space constraints into TinyMPC as well as giving it the capability to update its estimation based on known gate landmarks. Ideally, it will run fast enough to reliably perform drone racing tasks at up to $2\,\mathrm{m/s}$ in varying environments and gate orientations.

This version of the design targets the following deployment pipeline:

```text
PyTorch
    ↓
NEMO fake-quantized model
    ↓
NEMO integer-deployable model
    ↓
quantized ONNX
    ↓
DORY graph
    ↓
generated GAP8 C
    ↓
GVSOC
    ↓
AI-deck hardware
```

The model is considered successfully deployed only when the generated GAP8 implementation reproduces the quantized ONNX computation according to the numerical-equivalence requirements defined below.

# Constraints

## Hardware Constraints

Due to the limitations of the GAP8 chip, our CNN is constrained to the following specifications:

- fixed $160 \times 120 \times 1$ monochrome input;
- static batch size of one;
- INT8 weights and activations;
- a hard ceiling of approximately $180\,000$ parameters;
- a target below approximately $20$--$30$ million MACs;
- no dynamic allocation during inference;
- no dynamic tensor dimensions;
- bounded inference and communication latency compatible with the flight-control schedule.

Actual deployability depends heavily on activation tiling and operator support. The AI-deck has approximately $64\,\mathrm{kB}$ of cluster L1 and $512\,\mathrm{kB}$ of shared L2. We must carefully measure and restrict peak intermediate activation size.

Parameter count alone does not determine whether the network is deployable. The relevant implementation quantities are

- peak live activation memory;
- the number of simultaneously live input, output, skip, and weight tiles;
- L1 tile dimensions;
- convolution halo overhead;
- L2-to-L1 DMA traffic;
- generated GAP8 cycle count;
- controller deadline misses and watchdog behavior.

## Deployment and Training Constraints

Due to the nature of DORY and its NEMO frontend, we are limited in the operations that can appear in the deployed graph. Training-only teachers, losses, augmentations, and auxiliary heads may use more general operations as long as they are absent from the exported student graph.

### Allowed Operations

The version-one deployed graph is restricted to

- standard 2D convolution;
- $3 \times 3$ depthwise convolution;
- $1 \times 1$ pointwise convolution;
- BatchNorm during training;
- folded BatchNorm during deployment;
- ReLU or NEMO-compatible clipped activation;
- a final linear $1 \times 1$ convolution;
- optional single-wire residual addition only after an isolated deployment and parity test;
- static average pooling only if demonstrated to survive the complete NEMO--DORY pipeline unchanged.

### Banned Operations

The version-one deployed graph must not contain

- interpolation;
- resize;
- transposed convolution;
- general feature concatenation;
- dynamic slicing;
- dynamic reshape;
- LayerNorm;
- GroupNorm;
- GELU;
- SiLU;
- Mish;
- attention;
- softmax;
- sigmoid;
- vector normalization;
- recurrent state inside ONNX;
- arbitrary ONNX post-processing;
- multiple terminal output nodes.

An operation appearing in PyTorch or ONNX should not be considered supported until DORY parses it into the intended deployed computation and numerical parity is demonstrated.

## Coordinate and Tensor Conventions

The coordinate conventions must be fixed before generating labels. The simulation, Python decoder, STM32 decoder, and TinyMPC interface must use the same conventions.

### Image Convention

The training tensor uses NCHW layout:

$$
I_t \in \mathbb{Z}^{1 \times 1 \times 120 \times 160}.
$$

The image convention is

- pixel origin at the upper-left corner;
- $u$ increasing to the right;
- $v$ increasing downward;
- width $W=160$;
- height $H=120$;
- raw or deterministically normalized 8-bit grayscale input.

Any crop, mirror, vertical flip, distortion operation, or downsampling must be explicitly represented in the camera configuration and label generator.

### Body-Frame Convention

We use a body-fixed frame with

- $+x$ forward;
- $+y$ left;
- $+z$ upward.

Let $R_{wb}$ rotate vectors from the body frame into the world frame, and let $p_t^w$ be the current world-frame vehicle position.

### Output Convention

The network has exactly one output tensor:

$$
Y_t \in \mathbb{Z}^{1 \times 12 \times 15 \times 20}.
$$

The logical output channel assignment is

| Channels | Output |
| --- | --- |
| $0$--$3$ | Four ordered gate-corner heatmap score fields |
| $4$--$7$ | Four fixed-normal half-space offset fields |
| $8$--$11$ | Four corresponding confidence score fields |

The same channel indices must be generated into a shared constants header used by both the Python reference decoder and the STM32 implementation.

## Racing Requirements and Tradeoffs

In drone racing, the CNN can tolerate conservative false positives in obstacle avoidance more readily than false negatives. Conversely, gate detection can tolerate some missed detections while requiring a low false-positive rate.

The reason is that a gate landmark can appear in multiple consecutive frames, allowing additional opportunities for a reliable estimator update. Once a gate is correctly estimated, much of the accumulated state-estimation drift can be corrected. An obstacle can approach rapidly, and a falsely safe prediction can immediately lead to collision.

The safety output must therefore be evaluated primarily using

- false-safe rate;
- upper-tail clearance overestimation;
- collision rate;
- confidence calibration;
- performance at the target speed.

The gate output must be evaluated primarily using

- phantom-gate rate;
- corner localization error;
- PnP reprojection error;
- gate-pose error;
- completed-gate rate.

## Vision Challenges

One problem in gate detection is the occurrence of phantom gates: false positives produced by unrelated rectangular structures, shadows, wall edges, and obstacle outlines.

A gate candidate is accepted only when

- the required corner peaks exceed tunable score thresholds;
- the corner channels follow the required ordering;
- the resulting quadrilateral is convex;
- the side-length and diagonal ratios are physically plausible;
- the implied gate dimensions and pose are consistent with the calibrated camera;
- PnP reprojection error is below a threshold;
- the candidate is temporally consistent with recent detections and vehicle motion.

The initial implementation requires all four corners. A three-corner fallback may be considered later when one corner is occluded or outside the image, provided that the missing corner can be reconstructed from a known rectangular gate model and the resulting pose passes a stricter consistency test.

The gate opening and surrounding frame must be treated separately. A valid gate detection defines a candidate safe aperture. A safety constraint that intersects this opening may be relaxed only when

- gate confidence is high;
- PnP geometry is valid;
- the predicted trajectory passes through a shrunken version of the physical aperture;
- the gate estimate is temporally stable;
- no independent obstacle evidence occupies the opening.

# Architecture

Gate-corner detection requires enough spatial resolution to localize corners. We therefore use a $20 \times 15$ heatmap and recover sub-cell precision using a small local refinement outside the network.

Safe-corridor construction requires a wider receptive field and produces a small number of controller-relevant scalar offsets. Each offset also has a confidence score. When forced to choose between optimistic and conservative free-space estimates, the model should underestimate usable free space.

The network does not perform dense scene reconstruction. It does not output a depth map, inverse-depth map, semantic segmentation, or dense danger map. It extracts the minimum visual geometry needed by the controller.

Overall, the controller-facing representation is

$$
z_t=
\left[
\{H_j(u,v)\}_{j=1}^{4},
\{d_i,c_i\}_{i=1}^{4}
\right],
$$

where $H_j$ is an ordered corner heatmap score, $d_i$ is a conservative fixed-normal half-space offset, and $c_i$ is the confidence score associated with that offset.

## Network Topology

The version-one student uses a completely sequential depthwise-separable topology:

```text
160 × 120 × 1
        ↓
80 × 60 × 16
        ↓
80 × 60 × 24
        ↓
40 × 30 × 32
        ↓
40 × 30 × 48
        ↓
20 × 15 × 64
        ↓
20 × 15 × 96
        ↓
6 depthwise-separable blocks at 20 × 15 × 96
        ↓
1 × 1 linear convolution
        ↓
20 × 15 × 12
```

A proposed exact layer specification is

| ID | Operation | Kernel / stride / padding | Input | Output | Activation |
| --- | --- | --- | --- | --- | --- |
| L0 | Standard Conv | $3\times3/2/1$ | $160\times120\times1$ | $80\times60\times16$ | BN + ReLU/PACT |
| L1 | Depthwise Conv | $3\times3/1/1$ | $80\times60\times16$ | $80\times60\times16$ | BN + ReLU/PACT |
| L2 | Pointwise Conv | $1\times1/1/0$ | $80\times60\times16$ | $80\times60\times24$ | BN + ReLU/PACT |
| L3 | Depthwise Conv | $3\times3/2/1$ | $80\times60\times24$ | $40\times30\times24$ | BN + ReLU/PACT |
| L4 | Pointwise Conv | $1\times1/1/0$ | $40\times30\times24$ | $40\times30\times32$ | BN + ReLU/PACT |
| L5 | Depthwise Conv | $3\times3/1/1$ | $40\times30\times32$ | $40\times30\times32$ | BN + ReLU/PACT |
| L6 | Pointwise Conv | $1\times1/1/0$ | $40\times30\times32$ | $40\times30\times48$ | BN + ReLU/PACT |
| L7 | Depthwise Conv | $3\times3/2/1$ | $40\times30\times48$ | $20\times15\times48$ | BN + ReLU/PACT |
| L8 | Pointwise Conv | $1\times1/1/0$ | $20\times15\times48$ | $20\times15\times64$ | BN + ReLU/PACT |
| L9 | Depthwise Conv | $3\times3/1/1$ | $20\times15\times64$ | $20\times15\times64$ | BN + ReLU/PACT |
| L10 | Pointwise Conv | $1\times1/1/0$ | $20\times15\times64$ | $20\times15\times96$ | BN + ReLU/PACT |
| L11--L22 | Six repeated DS blocks | DW $3\times3/1/1$ + PW $1\times1/1/0$ | $20\times15\times96$ | $20\times15\times96$ | BN + ReLU/PACT after each convolution |
| L23 | Final Pointwise Conv | $1\times1/1/0$ | $20\times15\times96$ | $20\times15\times12$ | Linear |

This configuration is expected to require approximately $28.2$ million MACs and approximately $75\,000$ convolution weights. The exact values must be generated from the implemented model and checked into the repository. The parameter target is an upper bound, so a smaller network is acceptable when it meets the closed-loop accuracy requirements.

All convolutions followed by BatchNorm use `bias=False`. The final linear convolution may use a bias only after the exact NEMO and DORY export behavior has been validated.

Version one contains no residual connections. A residual variant may be evaluated later after an isolated residual block passes PyTorch-to-hardware parity testing.

## Shared Output Range

NEMO commonly uses a shared quantization scale for a layer output. Since the heatmap, offset, and confidence channels share one terminal tensor, all three channel groups should use comparable numerical ranges.

Let $S>0$ be the chosen output score limit, for example $S=6$. The training targets should occupy approximately $[-S,S]$.

For an offset $d_i \in [d_{\min},d_{\max}]$, define the encoded target

$$
s_i^d
=
2S\frac{d_i-d_{\min}}{d_{\max}-d_{\min}}-S.
$$

The STM32 decodes the spatially averaged output according to

$$
d_i
=
d_{\min}
+
\frac{\operatorname{clip}(\bar{s}_i^d,-S,S)+S}{2S}
\left(d_{\max}-d_{\min}\right).
$$

Corner and confidence outputs use the same approximate score range. Confidence is thresholded at zero, and heatmaps are decoded using relative peak scores. No probability conversion is required onboard.

## Gate-Corner Representation

The gate-corner output uses four ordered heatmap score channels:

1. top-left;
2. top-right;
3. bottom-right;
4. bottom-left.

Each channel is a $20\times15$ score field. The network does not contain sigmoid or softmax operations.

The post-processing procedure is

```text
four corner score fields
    → integer argmax per channel
    → optional 3 × 3 weighted centroid
    → heatmap-to-image coordinate conversion
    → peak and ambiguity checks
    → ordered quadrilateral checks
    → PnP
    → temporal filtering and state-estimator update
```

If $(\hat{u}_j^h,\hat{v}_j^h)$ is the refined heatmap coordinate, the corresponding image coordinate is

$$
\hat{u}_j
=
\frac{W}{20}
\left(\hat{u}_j^h+\frac{1}{2}\right)-\frac{1}{2},
$$

$$
\hat{v}_j
=
\frac{H}{15}
\left(\hat{v}_j^h+\frac{1}{2}\right)-\frac{1}{2}.
$$

The gate pose is recovered from known gate-frame centerline corner geometry using the calibrated camera model. This matches the real annotation convention: each labeled corner lies halfway between the inner opening and the outer gate edge. The physical free aperture is recovered separately from the known gate dimensions. The gate measurement should be rejected when the PnP reprojection error or implied pose lies outside the physically plausible range.

## Fixed-Normal Safe Corridor

Version one uses four fixed forward-facing body-frame directions:

$$
n_i
=
\begin{bmatrix}
\cos\theta_i\\
\sin\theta_i\\
0
\end{bmatrix},
$$

with an initial choice

$$
\theta_i
\in
\{-45^\circ,-15^\circ,15^\circ,45^\circ\}.
$$

The exact angles should be matched to the calibrated horizontal field of view and evaluated through ablation.

For each direction, the network predicts

$$
\{d_i,c_i\}_{i=1}^{4}.
$$

Here, $d_i$ is the conservative free distance available to the center of the drone along $n_i$. It is a sparse controller-specific geometric quantity and is not a dense depth estimate.

The body-frame constraint is

$$
n_i^\top \delta p_k^b
\le
d_i-\rho_i,
$$

where $\rho_i$ contains the physical and statistical safety margins.

The scalar values are decoded from the output fields by spatial averaging:

$$
\bar{s}_i^d
=
\frac{1}{300}
\sum_{u=1}^{20}
\sum_{v=1}^{15}
Y_{4+i}(u,v),
$$

$$
\bar{s}_i^c
=
\frac{1}{300}
\sum_{u=1}^{20}
\sum_{v=1}^{15}
Y_{8+i}(u,v).
$$

Spatial averaging is performed outside DORY. During training, a field-consistency loss encourages each scalar channel to remain approximately uniform over the $20\times15$ grid.

## Latency Optimizations

In order to reduce latency, we

- downsample early;
- reach $20 \times 15$ after three stride-2 stages;
- keep the remaining network at $20 \times 15$;
- use channel counts divisible by eight;
- avoid large live skip tensors;
- avoid concatenation;
- avoid decoder upsampling;
- use only $3\times3$ depthwise and $1\times1$ pointwise kernels after the stem;
- estimate layer-level L1 tile usage;
- record DMA bytes and cycle counts;
- measure the generated implementation instead of relying only on theoretical MACs.

The final architecture is accepted only when its worst-case inference latency, communication latency, and controller computation all fit within the flight-control timing budget without causing watchdog resets or starvation of other real-time tasks.

# Training

## Dataset Composition

The initial dataset will contain approximately $75\,000$ simulated monochrome images and several thousand images from the real track.

The synthetic dataset should be divided by scene and trajectory rather than by individual frame:

- $60\,000$ training frames;
- $7\,500$ validation frames;
- $7\,500$ held-out synthetic test frames.

Adjacent frames from the same trajectory must remain in the same split. Entire track layouts, lighting configurations, gate assets, obstacle arrangements, and randomization seeds should be held out for validation and testing.

The dataset should contain four broad scenario classes:

- gate-only scenes;
- obstacle-only scenes;
- scenes containing both a gate and obstacles;
- hard-negative scenes containing rectangular textures, shadows, frames, windows, and other structures that can resemble gates.

The sampling distribution should also be stratified by

- gate image size;
- gate distance;
- gate yaw, pitch, and roll;
- gate position in the image;
- number of visible corners;
- obstacle distance;
- obstacle bearing;
- free-corridor width;
- image brightness and contrast;
- vehicle translational and angular velocity.

Safety-critical examples near the decision boundary should be oversampled. These include small changes in pose that change one direction from safe to unsafe, partially occluded obstacles, narrow gate apertures, and obstacles visually aligned with a gate opening.

## Isaac Sim Scene Design for Sim-to-Real Transfer

The best scene-generation strategy is calibrated domain randomization. The simulator should first reproduce the measured real system as closely as practical. Randomization should then cover the remaining uncertainty around this calibrated baseline.

Pure photorealism can overfit to a small set of synthetic appearances. Unbounded randomization can produce a distribution that wastes model capacity on impossible scenes. The desired distribution contains realistic anchor scenes, controlled randomization around measured values, and a smaller set of deliberately difficult outliers.

A reasonable initial mixture is

- $60\%$ track-matched scenes using measured geometry, materials, and camera parameters;
- $30\%$ realistically randomized scenes;
- $10\%$ adversarial or unusual hard cases.

### Real-System Measurement Before Rendering

Before producing the main dataset, verify that we have the correct:

- HM01B0 camera intrinsics;
- lens-distortion coefficients;
- camera-to-body extrinsics;
- exact crop and readout mode;
- whether the firmware applies mirror or vertical-flip operations;
- exposure and gain settings used during flight;
- frame rate and row-readout behavior;
- grayscale range, black level, and clipping behavior;
- representative real-image brightness histograms;
- vignetting;
- fixed-pattern noise;
- motion blur at several translational and angular velocities.

The Himax HM01B0 is a monochrome rolling-shutter sensor with programmable exposure, analog gain, digital gain, black-level calibration, and QVGA/QQVGA operation. These sensor properties should define the randomization ranges instead of arbitrary generic camera noise.

### Rolling Shutter and Motion Blur

The HM01B0 uses an electronic rolling shutter. At high angular velocity, different image rows correspond to slightly different camera poses. This can shift gate corners and bend vertical edges.

If Isaac Sim does not directly reproduce the required rolling-shutter readout, approximate it by

1. simulating the camera trajectory during one frame;
2. rendering several temporal subframes;
3. assigning or warping row bands according to row capture time;
4. applying the measured exposure interval;
5. downsampling to the inference resolution.

Motion blur should be tied to the simulated translational and angular velocity. Isaac Sim supports motion-blur rendering using synchronized motion and temporal subframes. Blur should be calibrated using real flight images rather than chosen solely for visual appearance.

Frames should be generated from physically plausible trajectories, not only independently teleported cameras. The trajectory generator should include

- nominal gate approaches;
- lateral and vertical tracking error;
- yaw error;
- overshoot and recovery;
- motion up to and slightly beyond $2\,\mathrm{m/s}$;
- angular rates representative of racing turns;
- accelerations representative of TinyMPC commands.

### Geometry and Collision Modeling

The primary track environments should be built from measured geometry:

- exact gate inner width and height;
- gate-frame thickness and depth;
- gate support structures;
- wall, floor, and ceiling dimensions;
- obstacle dimensions;
- camera height and mounting angle;
- expected flight volume.

Gate meshes require separate geometry for the solid frame and the free opening. The collision geometry used to generate safe-corridor labels must agree with the physical volume occupied by the real frame.

Use several geometry families:

- exact replicas of the real track;
- perturbed versions of the real track;
- new held-out track layouts;
- cluttered variants containing visually confusing structures.

Randomize obstacle locations and shapes within physically plausible limits. Include boxes, panels, poles, cylinders, hanging objects, and large wall-like obstacles. The goal is to learn obstacle geometry and free motion rather than memorize one object category.

### Materials, Textures, and Lighting

Use Isaac Sim Replicator to randomize

- gate-frame albedo and roughness;
- floor and wall textures;
- obstacle materials;
- texture scale and rotation;
- light position;
- light intensity;
- light size and shadow softness;
- dome-light intensity and environment texture;
- localized shadows and partial illumination.

The ranges should be estimated from real track measurements and photographs. Since the sensor is monochrome, color variation matters through its effect on luminance and the sensor spectral response. Randomized RGB materials should therefore be converted through the same grayscale rendering and sensor pipeline used for training.

Important lighting cases include

- evenly illuminated scenes;
- dark corners;
- hard shadows crossing the gate;
- bright backgrounds behind a gate;
- backlighting;
- partial saturation;
- light flicker or banding consistent with the sensor exposure and local mains frequency.

Lighting randomization should preserve a substantial fraction of realistic, correctly exposed images. Excessive random darkness or saturation can teach the network to ignore useful contrast.

### Sensor-Domain Randomization

After rendering a linear or high-quality grayscale image, apply a calibrated sensor model. Randomize within measured ranges:

- exposure;
- analog and digital gain;
- black-level offset;
- gamma or tone curve, if present in the real pipeline;
- shot noise;
- read noise;
- row and column fixed-pattern noise;
- vignetting;
- slight defocus;
- motion blur;
- hot and dead pixels;
- 8-bit quantization and clipping.

Do not add JPEG compression when the deployed CNN receives raw camera bytes. Every augmentation should correspond to a plausible physical or firmware effect.

A useful fitting procedure is to compare real and synthetic distributions of

- mean intensity;
- intensity variance;
- gradient magnitude;
- edge width;
- high-frequency noise power;
- vignetting profile;
- saturation fraction.

Randomization parameters should be adjusted until synthetic images cover the real measurements without placing most samples far outside them.

### Camera-Pose and Calibration Randomization

Randomize small residual uncertainties around the calibrated camera:

- focal length;
- principal point;
- radial distortion;
- camera-to-body translation;
- camera-to-body roll, pitch, and yaw.

These perturbations should reflect measured calibration uncertainty and mounting repeatability. Large arbitrary intrinsic changes would conflict with the fixed PnP and controller geometry.

The label generator must always use the same randomized camera parameters that produced the image.

### Distractors and Hard Negatives

Phantom-gate prevention requires dedicated negative examples. Include

- rectangular wall decorations;
- door and window frames;
- shelving;
- shadows forming rectangles;
- overlapping poles;
- partially visible gates that do not satisfy the complete geometry;
- two unrelated objects whose corners approximately form a quadrilateral;
- solid panels with gate-like outer borders;
- gate frames whose openings are blocked.

The detector should learn that the four corners belong to one physically consistent gate aperture.

### Frame and Sequence Generation

Even though version one consumes one image at a time, data should be generated as short trajectories. This produces realistic correlations among

- pose;
- speed;
- motion blur;
- rolling shutter;
- gate scale change;
- obstacle approach rate.

Training may sample individual frames, while validation and closed-loop simulation should preserve temporal sequences.

Randomization should occur at multiple timescales:

- track layout and material randomization per episode;
- lighting randomization per episode or short sequence;
- exposure and gain changes slowly across frames;
- shot noise independently per frame;
- moving distractors according to physics.

### Real-Image Integration

Several thousand real images should be included through mixed synthetic-real training or final fine-tuning.

Real corner labels can be produced manually or with assisted annotation followed by human verification. Real safe-offset labels should come from measurable geometry whenever possible:

- external motion capture plus a measured track map;
- a temporary depth sensor used only for label collection;
- an offline multi-view reconstruction;
- a high-capacity teacher whose predictions are verified against geometry.

When the real track geometry and vehicle pose are known, compute the same inflated-obstacle ray distances used in simulation. This produces directly compatible labels.

Keep a held-out real test set that is never used to tune augmentation ranges or thresholds.

### Active Error Mining

After each deployment iteration, record images associated with

- false gate detections;
- missed gates;
- large corner error;
- unsafe clearance overestimation;
- low-confidence control decisions;
- abrupt output changes;
- near-collisions;
- lighting or motion regimes absent from the training set.

Add these examples to a dedicated hard-example set. Update the simulator to reproduce the missing factor when possible. This closes the loop between real failures and synthetic scene design.

## Label Generation

### Gate-Corner Labels

For both simulated and real data, a gate corner is defined on the centerline of the physical frame, halfway between the inner opening and the outer edge. The simulator may use the inner semantic hole to identify and associate a visible gate, but that temporary raster measurement must be replaced by the projected centerline geometry before generating the training heatmap. The gate asset's measured inner and outer dimensions must be stored with the scene metadata; they must not be inferred from image appearance.

Let the exact projected heatmap coordinate for corner $j$ be $(u_j^h,v_j^h)$. Generate a Gaussian target

$$
H_j^*(u,v)
=
S_h\exp\left(
-\frac{(u-u_j^h)^2+(v-v_j^h)^2}{2\sigma_h^2}
\right)
-
B_h,
$$

where $S_h$ and $B_h$ are selected so that the target occupies approximately the common output range $[-S,S]$.

Corner labels must be projected using the exact randomized camera model, distortion, crop, and downsampling operations.

For each corner, store a visibility mask

$$
m_j^{\mathrm{corner}}\in\{0,1\}.
$$

A corner is invalid when it is

- behind the camera;
- outside the image;
- occluded by geometry;
- too close to the clipping boundary for reliable annotation.

Frames with no valid gate use background corner targets.

### Safe-Offset Labels

Inflate every solid obstacle by the drone collision radius and an additional label margin. Let the resulting forbidden set in the body frame be $\mathcal{O}_{\mathrm{inflated}}$.

For fixed direction $n_i$, define the first collision distance

$$
d_i^*
=
\inf\left\{
\tau\ge0:
\tau n_i\in\mathcal{O}_{\mathrm{inflated}}
\right\}.
$$

When no obstacle occurs before the maximum sensing range, set

$$
d_i^*=d_{\max}.
$$

Clip all labels to

$$
d_i^*\in[d_{\min},d_{\max}].
$$

The gate opening is free space, while the gate frame is part of $\mathcal{O}_{\mathrm{inflated}}$.

For direction $i$, define a validity mask

$$
m_i^{d}\in\{0,1\}.
$$

A direction is invalid when the relevant geometry lies outside the modeled camera support, when the label is corrupted, or when a renderer/annotation consistency check fails.

The confidence target is positive for a valid observable direction and negative for an invalid direction. It should represent whether the offset is supported by the current image, not whether the direction is physically safe.

## Teacher-Student Distillation

We may use a larger teacher with privileged inputs such as

- metric depth;
- obstacle segmentation;
- surface normals;
- optical flow;
- multiple frames;
- exact simulator state.

The deployed student still receives one $160\times120$ grayscale image and outputs only the 12-channel tensor.

Useful training-only auxiliary tasks include

- dense obstacle/free-space segmentation;
- gate-frame segmentation;
- obstacle boundaries;
- dense depth or inverse depth;
- corner visibility.

All auxiliary heads are removed before NEMO conversion. Distillation may be applied to

- final corner heatmaps;
- safe-offset predictions;
- confidence predictions;
- selected intermediate feature tensors.

## Loss Function

The total loss is

$$
\mathcal{L}
=
\lambda_h\mathcal{L}_{\mathrm{heatmap}}
+
\lambda_d\mathcal{L}_{\mathrm{offset}}
+
\lambda_c\mathcal{L}_{\mathrm{confidence}}
+
\lambda_f\mathcal{L}_{\mathrm{field}}
+
\lambda_g\mathcal{L}_{\mathrm{geometry}}
+
\lambda_t\mathcal{L}_{\mathrm{distill}}.
$$

### Corner Heatmap Loss

Use focal-MSE or focal logistic loss on the four corner score fields:

$$
\mathcal{L}_{\mathrm{heatmap}}
=
\frac{1}{4HW}
\sum_{j=1}^{4}
\sum_{u,v}
w_j(u,v)
\left(H_j(u,v)-H_j^*(u,v)\right)^2.
$$

The weights $w_j$ increase the contribution of pixels close to the true corner and hard false peaks.

A training-only coordinate loss may also be computed from a differentiable soft-argmax. This operation is used only in the loss and does not appear in the exported graph.

### Offset Loss

Decode the training prediction from the mean field and the affine output mapping. Let

$$
e_i=\hat d_i-d_i^*.
$$

Use an asymmetric Huber loss:

$$
\mathcal{L}_{\mathrm{offset}}
=
\frac{
\sum_i m_i^d
\left[
\lambda_{+}\mathbf{1}_{e_i>0}\operatorname{Huber}(e_i)
+
\lambda_{-}\mathbf{1}_{e_i\le0}\operatorname{Huber}(e_i)
\right]
}{
\sum_i m_i^d+\epsilon
},
$$

with

$$
\lambda_{+}>\lambda_{-},
$$

because overestimating free space is more dangerous than underestimating it.

### Confidence Loss

Use binary cross-entropy with logits or an equivalent signed-score loss:

$$
\mathcal{L}_{\mathrm{confidence}}
=
\frac{1}{4}
\sum_i
\operatorname{BCEWithLogits}(\bar{s}_i^c,c_i^*).
$$

Class weights should be chosen from the observed valid/invalid ratio.

### Field-Consistency Loss

The offset and confidence channels are decoded by spatial averaging. Encourage each scalar field to be spatially uniform:

$$
\mathcal{L}_{\mathrm{field}}
=
\sum_{i=1}^{4}
\operatorname{Var}_{u,v}\left(Y_{4+i}(u,v)\right)
+
\sum_{i=1}^{4}
\operatorname{Var}_{u,v}\left(Y_{8+i}(u,v)\right).
$$

### Gate-Geometry Loss

A training-only geometry loss can penalize inconsistent recovered corners. Examples include

- non-convex quadrilaterals;
- opposite sides with inconsistent orientation;
- large PnP reprojection error;
- implausible gate aspect ratio after projection.

This term should be introduced after the basic heatmap detector is stable.

## Training Schedule

A recommended schedule is

1. train the deployment-compatible architecture in FP32;
2. verify corner and offset label generation independently;
3. train with moderate calibrated randomization;
4. introduce the full randomization and hard-negative curriculum;
5. add teacher distillation or training-only auxiliary losses;
6. evaluate on held-out synthetic scenes and a held-out real set;
7. convert to NEMO fake quantization;
8. fine-tune with QAT while monitoring decoded controller outputs;
9. freeze BatchNorm statistics;
10. convert to the integer-deployable model;
11. export quantized ONNX;
12. run numerical-equivalence tests before generating flight firmware.

During QAT, report task metrics after decoding the fake-quantized output. A low training loss with degraded quantized offsets or corner coordinates is considered a failed run.

# Quantization and Deployment

## Quantization Contract

The repository must contain a machine-readable quantization manifest recording, for every deployed tensor,

- tensor name;
- tensor shape;
- signedness;
- bit width;
- scale or shift;
- zero point, when applicable;
- clipping range;
- accumulator width;
- requantization multiplier and shift;
- output interpretation.

The final output uses a common bounded score representation so that all 12 channels retain useful precision under a shared output scale.

## Numerical-Equivalence Tests

Create a fixed golden set containing at least 100 images covering

- empty scenes;
- centered gates;
- off-axis gates;
- partially visible gates;
- hard phantom-gate distractors;
- close and distant obstacles;
- narrow corridors;
- low-light and high-contrast scenes;
- motion blur;
- all-zero input;
- all-maximum input;
- synthetic checkerboard and impulse patterns.

For each input, save and compare

1. FP32 PyTorch output;
2. NEMO fake-quantized output;
3. NEMO integer-deployable output;
4. ONNX Runtime output;
5. DORY intermediate output where available;
6. GVSOC output;
7. physical GAP8 output.

Compare both

- raw integer output tensors;
- decoded corner coordinates, offsets, and confidence scores.

The deployment is blocked when any stage introduces an unexplained

- transpose;
- channel permutation;
- padding difference;
- signedness difference;
- scale difference;
- saturation;
- omitted operation;
- changed output layout.

## Layer-Bisect Diagnostic

The export pipeline must support a diagnostic model that exposes selected intermediate tensors. When final outputs disagree, compare stages layer by layer and identify the first divergent layer before changing the model or retraining.

## DORY Graph Audit

Every exported model must generate an audit report containing

- ONNX opset;
- node list and node order;
- tensor names and shapes;
- DORY-recognized layer sequence;
- BatchNorm and activation fusion status;
- generated tile dimensions;
- peak L1 and L2 estimates;
- estimated MACs and cycles;
- output type and memory layout;
- every warning emitted by NEMO, ONNX, and DORY.

The implementation agent must stop when DORY drops, neglects, or rewrites an operation in a way that changes the intended computation.

## Reproducible Environment

Pin and archive

- Python version;
- PyTorch version;
- NEMO commit;
- ONNX version;
- DORY commit;
- OR-Tools version;
- GAP SDK commit;
- PULP-NN commit;
- compiler version;
- model-export script;
- generated constants header;
- representative golden inputs and outputs.

A container or equivalent immutable environment should be checked into the project.

# How the Controller Will Avoid Obstacles

We take a carrot-and-stick approach. The carrot modifies the tracked reference to encourage motion toward a direction with usable free space. The stick adds conservative half-space constraints that prevent TinyMPC from selecting states beyond the predicted local boundary.

## Decoding the Safe Corridor

For each direction $i$, decode

$$
(d_i,c_i).
$$

Accept the offset when

$$
c_i\ge c_{\min}.
$$

Low confidence must never be interpreted as free space. Low-confidence behavior may include

- retaining a recent valid constraint for a short bounded interval;
- shrinking the commanded speed;
- using a conservative default offset;
- hovering or stopping when no direction is reliable.

The effective constraint offset is

$$
\tilde d_i
=
d_i-\rho_i,
$$

where

$$
\rho_i
=
r_{\mathrm{drone}}
+m_{\mathrm{tracking}}
+m_{\mathrm{latency}}
+m_{\mathrm{perception}}(c_i).
$$

The perception margin increases as confidence decreases.

## Transforming a Body-Frame Plane into the World Frame

The body-frame constraint is

$$
n_i^\top\delta p_k^b\le\tilde d_i.
$$

Let

$$
n_i^w=R_{wb}n_i.
$$

Since

$$
\delta p_k^w=p_k^w-p_t^w,
$$

the equivalent world-frame constraint is

$$
(n_i^w)^\top(p_k^w-p_t^w)
\le
\tilde d_i,
$$

or

$$
(n_i^w)^\top p_k^w
\le
\tilde d_i+(n_i^w)^\top p_t^w.
$$

Thus, for a position-only TinyMPC row,

$$
a_{p,i}=n_i^w,
$$

$$
b_i^w=\tilde d_i+(n_i^w)^\top p_t^w.
$$

The initial implementation sets

$$
a_v=0.
$$

A later implementation may include a velocity-dependent lookahead margin.

## Knot Activation

The same visual constraint need not be active at every horizon knot. For nominal knot $p_k^{\mathrm{nom}}$, compute

$$
g_{i,k}
=
(n_i^w)^\top
\left(p_k^{\mathrm{nom}}-p_t^w\right)
-
\tilde d_i.
$$

Activate plane $i$ at knot $k$ when

$$
g_{i,k}>-\epsilon_{\mathrm{activation}}.
$$

This applies the constraint to knots that approach the predicted boundary and avoids unnecessarily constraining the entire horizon.

## Reference Adjustment

Let $r_g^b$ be the preferred body-frame direction toward the current gate or waypoint, and let $n_{\mathrm{prev}}$ be the previously selected avoidance direction.

Score each safe direction using

$$
S_i
=
w_d\hat d_i
+w_g n_i^\top r_g^b
+w_h n_i^\top n_{\mathrm{prev}}
-w_vJ_{\mathrm{dynamic},i},
$$

where

- $\hat d_i$ is the confidence-adjusted free distance;
- $n_i^\top r_g^b$ rewards progress toward the goal;
- $n_i^\top n_{\mathrm{prev}}$ discourages left-right chattering;
- $J_{\mathrm{dynamic},i}$ penalizes directions that are difficult to reach from the current velocity.

Choose

$$
i^*=\arg\max_i S_i.
$$

Shift one or more reference knots toward $n_{i^*}$:

$$
p_k^{\mathrm{ref,new}}
=
p_k^{\mathrm{ref,nom}}
+\eta_k n_{i^*}^w,
$$

where $\eta_k$ increases smoothly over the near horizon and is limited by vehicle dynamics and corridor width.

The reference shift provides the incentive to move around an obstacle, while the active half-spaces provide the local boundary.

## Gate Passage and Constraint Relaxation

A detected gate defines a planar aperture. Let $c_g$, $u_g$, $v_g$, and $n_g$ denote the gate center, horizontal axis, vertical axis, and normal.

At a selected crossing knot $k_g$, enforce the shrunken aperture

$$
-\left(\frac{w_g}{2}-r_{\mathrm{safe}}\right)
\le
u_g^\top(p_{k_g}-c_g)
\le
\left(\frac{w_g}{2}-r_{\mathrm{safe}}\right),
$$

$$
-\left(\frac{h_g}{2}-r_{\mathrm{safe}}\right)
\le
v_g^\top(p_{k_g}-c_g)
\le
\left(\frac{h_g}{2}-r_{\mathrm{safe}}\right).
$$

A corridor plane may be relaxed within this aperture only when the full gate-validation logic passes. The solid gate frame remains an obstacle.

## Safety Interpretation

TinyMPC enforces the supplied half-space relative to its model, solver tolerance, and iteration budget. Physical collision safety additionally depends on

- perception accuracy;
- confidence calibration;
- state-estimation accuracy;
- camera and inference latency;
- model mismatch;
- constraint age;
- controller feasibility;
- ADMM convergence.

The phrase “hard constraint” refers to the optimization problem after the visual plane has been supplied. It is not an unconditional guarantee about the physical environment.

# Evaluation

## Offline Perception Metrics

### Gate Metrics

Report

- corner RMSE in input-image pixels;
- corner $95$th-percentile error;
- four-corner detection recall;
- phantom-gate rate per frame and per minute;
- PnP translation and rotation error;
- PnP reprojection error;
- confidence calibration.

### Safe-Corridor Metrics

Report

- mean absolute offset error;
- overestimation mean and $95$th/$99$th percentiles;
- false-safe rate at each direction;
- direction-selection accuracy;
- confidence expected calibration error;
- temporal output variation on real sequences.

### Quantized Metrics

Report all task metrics for

- FP32 PyTorch;
- fake-quantized NEMO;
- integer-deployable NEMO;
- ONNX Runtime;
- GVSOC;
- physical GAP8.

## Closed-Loop Metrics

Evaluate at increasing speeds:

$$
0.5,\ 1.0,\ 1.5,\ 2.0\ \mathrm{m/s}.
$$

Report

- collision rate;
- completed-gate rate;
- track completion rate;
- minimum obstacle clearance;
- control-cycle latency;
- CNN inference latency;
- UART latency;
- TinyMPC solve time;
- ADMM residual and iteration count;
- infeasible solves;
- watchdog events;
- energy per inference.

# Ablation Plan

The following ablations should be evaluated after the baseline deploys correctly:

- four versus eight fixed directions;
- alternative direction angles;
- $20\times15$ versus a higher-resolution corner output;
- number of $20\times15\times96$ depthwise-separable blocks;
- confidence head enabled versus disabled;
- symmetric versus asymmetric offset loss;
- clean simulation versus calibrated domain randomization;
- sensor randomization enabled versus disabled;
- rolling-shutter and motion-blur modeling enabled versus disabled;
- teacher distillation enabled versus disabled;
- training-only segmentation or depth supervision enabled versus disabled;
- synthetic-only versus mixed synthetic-real training;
- FP32 versus QAT;
- residual block after isolated DORY validation.

# Test Progression

1. Verify label-generation geometry using rendered debug overlays.
2. Train and validate the FP32 model.
3. Run QAT and decoded-output evaluation.
4. Pass ONNX and DORY graph audits.
5. Pass ONNX-to-GVSOC parity.
6. Pass GVSOC-to-GAP8 parity.
7. Run camera streaming with motors disabled.
8. Test static gates and obstacles by hand-carrying the drone.
9. Hover with the visual constraints disabled.
10. Enable monitoring-only visual predictions.
11. Test low-speed stopping before a frontal obstacle.
12. Test low-speed reference deflection around an obstacle.
13. Test low-speed gate detection and estimator updates.
14. Test combined gate passage and obstacle avoidance.
15. Increase speed gradually toward $2\,\mathrm{m/s}$.

# Required Deliverables

The training and deployment agent must produce

- exact PyTorch architecture source;
- architecture manifest with every tensor shape;
- parameter and MAC report;
- training configuration and random seeds;
- dataset manifest and scene split;
- Isaac Sim generation configuration;
- label-generator commit;
- FP32 checkpoint;
- fake-quantized checkpoint;
- integer-deployable checkpoint;
- exported ONNX;
- quantization manifest;
- ONNX and DORY graph audit;
- pinned dependency versions and commits;
- generated GAP8 C source;
- GVSOC logs;
- golden input/output vectors;
- layer-bisect diagnostic procedure;
- GAP8 latency, memory, DMA, and energy measurements;
- decoded task metrics at every deployment stage;
- closed-loop flight-test results.

A statement that the network runs on GAP8 is insufficient. The agent must demonstrate that the generated implementation computes the intended quantized network and that the decoded controller outputs retain their physical meaning.
