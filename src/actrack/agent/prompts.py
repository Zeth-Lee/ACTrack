"""VLM prompts for the two selectable language protocols.

Chinese is the original runtime prompt. English is its faithful translation,
with source names Tracker -> Matching and SAM -> Motion. The asymmetric
confidence, similar-object, occlusion and whole/part rules are preserved.
Only one language is sent per request; the controller uses Tracker/SAM internally.
"""

VLM_REPROMPT_PROMPT_ZH = (
    "图片的左上角是确定的首帧追踪物，首先，你需要确定它是个什么物体，并保证一个准则——"
    "追踪的物体类型一定要一致，然后请你注意它的外观而非动作，作为接下来选择的依据，"
    "右上角为当前帧的全帧图片，蓝色框中是SAM追踪的bbox，绿色框是Tracker追踪的bbox，用于把控整体，"
    "下边的两张图片则是我裁切出的Tracker和SAM的裁切图，用于观察具体细节。\n"
    "现在，我需要你分析Tracker和SAM谁的选择是正确的，你需要结合四张图片你获取到的信息，"
    "做出最置信的选择，你可以选择“SAM”，“Tracker”和“Uncertain”，注意，Tracker需要你完全确定这是一个清晰的选择，"
    "如果有任何一点不清晰就要选择Uncertain，Tracker必须高置信才可以选择，不要有多余回答。\n"
    "注意：\n"
    "1.不要把动作和位置的变化作为参考因素，因为在一段长视频中，跟踪物和干扰物随时可以出现运动变化，"
    "所以如果是同类型物体，并且自身没有外观上的显著特征（如颜色，花纹等），那你就需要回答“Uncertain”\n"
    "2.请注意场景当中出现多个类似物体时，往往是SAM更能追踪到正确的物体上，所以如果你认为SAM和Tracker"
    "的特征与首帧GT的差异并不是非常显著，你需要选择“Uncertain”\n"
    "3.请注意SAM是否像是被遮挡了，因为当SAM被遮挡时，Tracker有可能漂移到类似物体上，你需要警惕这种情况，"
    "如果SAM的框有任何一点像是被遮挡的情况，请回复“Uncertain”\n"
    "4.有些情况是SAM没有分清一个物体的整体和局部，当你发现SAM和Tracker框的是同一个位置，"
    "区别只在于是否框住了尾巴等内容时，你可以直接选择“Tracker”，但是前提是一定要保证二者框住的是同一个物体"
)


VLM_REPROMPT_PROMPT_EN = (
    "The top-left panel shows the confirmed tracking target from the first frame. First, identify "
    "what kind of object it is and make sure that one principle is followed: the type of object being"
    " tracked must remain the same. Then focus on its appearance rather than its movements as the "
    "basis for the following choice. The top-right panel shows the full current frame. The blue box "
    "is the bounding box tracked by Motion, and the green box is the bounding box tracked by "
    "Matching; use this view to understand the overall scene. The two images below are the crops I "
    "extracted for Matching and Motion, for examining specific details.\n"
    "Now, I need you to analyze which of Matching and Motion has made the correct choice. Combine the"
    " information obtained from all four images to make the choice you are most confident in. You may"
    " choose \"Motion\", \"Matching\", or \"Uncertain\". Note that you must be completely sure that "
    "Matching is a clear choice before selecting it. If anything is unclear, choose Uncertain. Only "
    "select Matching when you have high confidence. Do not provide any additional response.\n"
    "Notes:\n"
    "1. Do not use changes in movement or position as reference factors, because the target and "
    "distractors can change their motion at any time in a long video. Therefore, if they are objects "
    "of the same type and lack distinctive appearance features (such as color or patterns), you need "
    "to answer \"Uncertain\".\n"
    "2. Note that when multiple similar objects appear in the scene, Motion is often better able to "
    "track the correct object. Therefore, if you think that the features of the Motion and Matching "
    "candidates do not differ very markedly from those of the first-frame ground-truth target, you "
    "need to choose \"Uncertain\".\n"
    "3. Check whether the target tracked by Motion appears to be occluded. When the target tracked by"
    " Motion is occluded, Matching may drift to a similar object, so be alert to this possibility. If"
    " there is even the slightest sign that the target in the Motion box may be occluded, reply "
    "\"Uncertain\".\n"
    "4. In some cases, Motion does not distinguish the whole object from a part of it. When you find "
    "that Motion and Matching box the same location and differ only in whether they include the tail "
    "or other such parts, you may directly choose \"Matching\", but you must first be certain that both"
    " boxes correspond to the same object."
)
