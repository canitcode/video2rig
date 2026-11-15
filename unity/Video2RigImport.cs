// video2rig: Unity's import settings for the FBX files blender/export_fbx.py writes (docs/GAME.md). Copy this file into
// an Editor folder of your project; it applies to every .fbx under a folder named Video2Rig.
using System;
using System.IO;
using UnityEditor;
using UnityEngine;

public class Video2RigImport : AssetPostprocessor
{
    const string Folder = "/video2rig/";

    // Keyframe reduction within these errors (degrees, percent) leaves the motion as the fit made it; Unity's own
    // default (Optimal, 0.5 / 0.5 / 0.5) moved joints and added shake on fast moves (docs/GAME.md, measured).
    public const float RotationError = 0.1f, PositionError = 0.1f, ScaleError = 0.1f;

    [Serializable]
    class Sidecar
    {
        public string root;
    }

    bool Mine => assetPath.EndsWith(".fbx", StringComparison.OrdinalIgnoreCase) &&
                 ("/" + assetPath.Replace('\\', '/')).ToLowerInvariant().Contains(Folder);

    void OnPreprocessModel()
    {
        if (!Mine)
            return;
        var importer = (ModelImporter)assetImporter;
        importer.animationType = ModelImporterAnimationType.Generic;
        importer.importAnimation = true;
        importer.animationCompression = ModelImporterAnimationCompression.KeyframeReduction;
        importer.animationRotationError = RotationError;
        importer.animationPositionError = PositionError;
        importer.animationScaleError = ScaleError;
        // the bone that carries the body's place (export_fbx.py writes it beside the FBX): Unity's root motion node,
        // so a clip that travels (reader_fit "travel") moves the character when the Animator applies root motion
        var side = Path.ChangeExtension(assetPath, ".v2r.json");
        if (File.Exists(side))
        {
            var root = JsonUtility.FromJson<Sidecar>(File.ReadAllText(side)).root;
            if (!string.IsNullOrEmpty(root))
                importer.motionNodeName = root;
        }
    }

    void OnPreprocessAnimation()
    {
        if (!Mine)
            return;
        var importer = (ModelImporter)assetImporter;
        var clips = importer.defaultClipAnimations;
        foreach (var clip in clips)
        {
            // a clip whose name ends in -loop (reader_fit "loop") loops; its end is already its start, so Unity's
            // loop pose (a blend of the two ends) stays off
            clip.loopTime = clip.name.EndsWith("-loop", StringComparison.Ordinal);
            clip.loopPose = false;
            clip.keepOriginalOrientation = true;
            clip.keepOriginalPositionY = true;
            clip.keepOriginalPositionXZ = true;
        }
        importer.clipAnimations = clips;
    }
}
