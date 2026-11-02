# Troubleshooting

**"PyTorch not found"** - install PyTorch with CUDA into the Python you run the tool with, or set
`V2R_TORCH_PYTHON` to a Python that has it (README, "Install").

**Tracking is very slow (10+ minutes) or "CUDA out of memory"** - a 5 s clip peaks at about 6.5 GB of video memory
at full size. Close other programs that use the card (games, Blender with a viewport, a game engine recording a
movie, another tracker): they also share the card's time, and a busy card makes a 1 minute tracking take 10+.
On Windows, when a program overflows the card's memory the driver silently moves it to system memory and everything
crawls: the tracker takes at most the card's memory minus 1.2 GB (`V2R_GPU_MEM_GB` sets it), and when that is not
enough it tracks again at 512 px by itself. Lower `track_side` to 512 to start there, or run with `--cpu` (slow but
works). Two clips started at once take turns on the card (a lock in the tool's cache folder).

**Tracking was fast and now takes 10x longer** - a laptop card that runs hot for a long time slows itself down:
`nvidia-smi --query-gpu=clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv` showed 270 MHz
instead of 3 GHz at 89 C (reason 0x20, thermal) after hours of tracking. Let it cool, raise the fan profile, track
fewer clips in a row.

**Frame 0 outline match below 0.85** (printed at the start) - the video's first frame is not your start frame: a
different pose (the head already turned), a different model, or a camera the generator moved a lot. Check
`work/<clip>/cameras-aligned.json` and the start frame. Points are only seeded where render and video agree, so a
small difference is fine.

**The report lists warnings** - `warnings` in `work/<clip>/report.json` says in plain words what to check: a weak
frame 0, frame ranges where the outline does not match (watch these in the review film), a joint that jumps in one
frame, the body under the floor. `none` means no number looks wrong; still watch the film.

**The body stays the wrong side up after a roll** (the log says "every lost run keeps its own turn") - the outline
did not prove the turn: with one view a roll is often invisible. Use two views (side and top see a bug on its back
at once). `solve.rescue_turn_cost` lower (0.1) lets the body turn for a smaller gain in outline.

**Legs or arms swap in the result** - use two views (front | side). With one view, make the prompt keep the limbs
apart and visible.

**A limb sticks inside the body or stays behind** - run with `--refine`. If the video itself hides the limb for
long, nothing can measure it: generate a clip with a second view.

**Hands or arms tremble** - the joint reader reads every frame on its own, and on small parts its error from one
frame to the next is several times the trunk's. A reader trained longer (`--steps 9000`) reads steadier. The fit's
smoothness can be raised for a calm clip: `python -m video2rig.reader.fit my_troll stagger --set w_acc=0.05` (it
writes `out/stagger-reader.npz` for `import_motion.py`); too much and fast moves (a strike, a landing) lose their
reach, so watch the review film. With the older road (`--old`): `solve.smooth_legs` (limbs) and `solve.smooth_body` in
`creature.json`, higher = smoother.

**The model goes through the floor / feet hover** - check that the start pose stands on the floor in the start frame.
Feet in generated clips slide a little, and the tool keeps what the video shows; pin them in the engine or with IK
keys.

**The Action plays too fast or too slow** - the timing is measured from motion; on a clip that barely moves it can
guess wrong. Force it: `"fps": "24"` (or `"30-5"`, `"25-25"`) in `creature.json` and delete `work/<clip>/`.

**Something looks wrong in Blender but not in the review film** - make sure you import into the same file you
exported from, with the armature in the same place, and that the scene's frame rate is the one you want (keys are
placed at the clip's real times).

**The auto-rig named the sides wrong or found no limb pairs** - the model must stand along X or Y. A model turned at
an angle (a deer at 25 degrees) is mirror-symmetric about neither axis: rotate it in Blender, apply, and run
`autorig_prepare.py` again.

**The auto-rig gave a leg that does not follow the mesh** - `joints.json` says which UniRig draw was kept; a floor
contact no draw reached gets a leg from the mesh's centre line (the log says "floor contact without a leg"). Run
`python -m video2rig.autorig.unirig <work> --draws 6` for more draws.

**A many-legged creature got too few legs** - each leg must reach the floor and leave the body as a thin tube. Legs
fused into a solid skirt with the body, or tucked under it, are not traced: separate or lower them in the model.

**A limb bone moves nothing (empty in `weights.json`)** - the mesh piece around it is a separate loose part that
lies against another limb. Join it to the body in Blender (or merge by distance) and run `autorig_build.py` again.

**One arm stands still in the fit while the video swings it** (the far arm in a side panel) - a side panel cannot
tell the twins apart well, and the far arm, behind the body, may be read where the near one is. Start the clip from the
colour suit (`blender/render_markers.py`, every limb its own colour) so the reader tells them apart. Still wrong: the other panel did not read the arm
surely either (its confidence in `work/<clip>/reader.npz`); retrain the reader longer, or film the move with the view
that sees that arm.

**The two panels disagree on how far the body went** - the generator drew the move a little differently in each
panel; the fit takes the side view's heights first and the front view's left and right. Much more than a few
percent of the height (`python -m video2rig check` fails the clip): generate it again.
