# How it works

## The idea

Frame 0 of a generated clip is our own render: the pose and the cameras are known exactly. So before any clip, the
tool can learn what this one creature looks like through these cameras, with exact labels and nothing drawn by
hand. It renders the creature in thousands of random poses and trains a small network, the joint reader, to find
every joint in such a picture. Then it reads every joint of a clip, in every frame and every view, and fits the rig
to those joints.

1. **Train a reader** (once per creature: `python -m video2rig.reader.prepare`, docs/WORKFLOW.md step 2b).
   - 3000 random poses of the rig within its joint ranges: the body turned, shifted or placed anywhere the start
     frame leaves room, on the floor or in the air, the jaw open in half of them, two limb ends brought together or
     one laid on the body. In about half of them the bones move during the frame, so the render has motion blur,
     sometimes with the whole body lunging or spinning. No animation goes in: the reader learns the creature, not a
     move, so one reader reads every move.
   - Blender renders every pose through the start frame's own cameras. The tool works out which joints each picture
     shows (the posed mesh in a depth buffer): a far leg behind the near one or an elbow behind the chest is
     hidden, and the reader is taught to say "not here" there instead of guessing.
   - The network: DINOv2-small (Meta, Apache 2.0) sees the picture at 448 px, a small decoder draws one heat map per
     joint. Every training picture is damaged the ways a generated frame differs from our render: re-framed, other
     colours and light, blur, a fast limb smeared into a see-through streak (Seedance draws a strike as a 30-100 px
     smear), lower resolution, noise, a shadow and a tinted floor. It is never mirrored: left and right are exactly
     what it must tell apart.
   - 5% of the poses are held out. Their error in pixels says whether the reader can be trusted (2-4 px for the
     readers in use). Longer training helps up to about 9000 steps.
2. **Prepare the clip.** Frames, the creature's silhouette in every frame, the content frame rate (Seedance clips
   drop frames, so one step in four can carry twice the motion), and the cameras aligned on frame 0 (Seedance
   re-frames the start image a little: scale, offset, sometimes a turn of 10+ degrees).
3. **Read.** The reader finds every joint in every frame and view, and says how sure it is (the peak of its heat
   map: about 1 where it sees the joint plainly, low where it guesses). It reads every picture 8 times, each copy
   shifted by a few pixels and scaled a little, brings the copies' heat maps back onto the picture and averages them:
   a reader that reads each frame on its own errs from frame to frame most on small, fast parts (hands, elbows,
   paws), and an error that does not repeat between the copies shrinks in their average (the arms' frame-to-frame
   error -34 to -47% on 30 clips). Half a minute on the graphics card for a 5 s clip. A prop is
   read as a few points of its own (a log's ends, a round stone's middle) and fitted as a rigid body.
4. **Panels in step.** A video model sometimes plays one panel ahead of the other. Both panels see a joint's
   height, so the two panels are matched frame by frame on their joints' heights (dynamic time warping); when that
   cuts how far they disagree by 15% or more, the front panel's readings are moved onto the side panel's frames.
5. **Fit, the whole clip at once.** One optimisation over all frames together finds the pose of every frame whose
   joints land where the reader saw them in every view: each reading weighted by how sure the reader was, within the
   rig's joint ranges (knees and elbows bend one way), smooth from frame to frame (the joints' acceleration and the
   angles' change from frame to frame cost a little), a light pull toward the start pose, and the feet not under the
   floor. It starts from the clip itself: every joint placed in 3D where its two readings meet, the body turned by
   the points on its root bone, every bone pointed at its next joint. A miss counts robustly, its scale narrowing
   from 40 px to 6 px: at first a joint far from its reading still pulls the pose out of a wrong start, at the end a
   wrong reading barely pulls. Heights come mostly from the side view; the front view foreshortens a lifted leg and
   mainly gives left and right. Left and right come from the reader as read: start the clip from the colour suit
   (`blender/render_markers.py`, every limb its own colour) and it does not mix them up. A jaw the reader sees opens for a bite. About a minute for a 5 s clip, on one processor
   thread, so a clip gives the same motion on every computer.

Nothing is added after the fit: no filter, no pinned feet, no travel by steps, no springs. The motion is what the
clip shows and the reader reads. A walk Seedance draws on the spot stays on the spot; give it travel in the game
(root motion from the standing feet) if it needs any.

Then the motion is written for Blender: every bone's rotation at the clip's real times, the root's position and every
prop's matrix.

On nine real clips of a turkey, against the same readings, this fit lands the keypoints closer to the side view's sure
readings than the earlier frame-by-frame fit with its clean-up passes did, in all nine, and they shake less in all
nine (1.1 to 13 times). On calm clips it misses the front view's sure readings by up to 2 px more: their heights
count a quarter.

Before the fit, `python -m video2rig check` uses the same reading to measure how far the two views disagree. When
the generator drew a different motion in each view (the body more than 1.5% of its height apart), no fit can make one
true motion out of them: generate the clip again.

## What it can and cannot see

- Two views see depth; one view guesses it from the start pose and the skeleton. Use two views for arms crossing
  the body, rolls, turns towards the camera.
- A limb behind the body in one view is read from the other view. Hidden in both, it is guessed, and those guesses
  are the reader's weakest point (a limb behind the body in the side view: 5-12 px off in our tests).
- The tool reproduces the video, including its mistakes: a sliding foot, a floating jump, a prop changing size, two
  views that do not agree.
- Fingers are not fitted (480p cannot resolve them); they keep the start pose. Curl them in Blender if needed.
- A held weapon's direction is not read: the hand's twist leaves no mark on the wrist and hand-tip points, so the
  blade can point the other way than in the video.
- A reader knows one creature through one set of cameras. A new model, rig or start frame needs a new reader.

## The older road (`--old`)

Before the reader, the tool followed points instead of joints. `run` still takes this road when the creature has no
`reader/reader.pt`, or with `--old`. Seeds on the model's visible pixels in the start frame (dense on limbs and
props) are followed through the clip by BootsTAPIR (Google DeepMind, Apache 2.0). Then one optimisation over all
frames finds the joint angles that put every tracked point on its track, while the model stays inside the video's
silhouette, joints stay in range and the motion stays smooth; when a body rolls over and the outline is lost, it
tries the body turned over. It needs no training, but a clip takes 1-3 minutes of tracking and 5-15 of solving, it
has many hand settings (`solve.*`, docs/CONFIG.md), and on the bench's wolf gallop it was 21 degrees per bone off.

## Why not the usual ways

- Fitting the outline frame by frame (what we did first): the outline of a near leg and a far leg is the same, so
  legs swapped and arms twisted far past what a joint can do; every clip needed hours of hand fixes.
- Pose detectors (2D keypoints): made for people, swap left and right in side views, nothing for animals. The joint
  reader is a pose detector too, but trained on this one creature through these cameras, with exact labels.
- Video mocap services: humans only, and they lose arms hidden by a prop.
