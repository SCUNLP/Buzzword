urban_direct_prompt = "Complete this task in Chinese:\nWord: {}\nExample: {}\nMeaning:"

urban_instruction_prompt = """
Instruction: Given a word or a phrase, use the provided example to infer the meaning of the word or the phrase. When analyzing the usage examples, interpret the context literally and think through it carefully to infer the meaning of the word (or the phrase). If the meaning cannot be inferred from the example, it is appropriate to say, "The meaning is unclear."
Word: {word}
Example: {example}
Analysis:
Meaning:
"""
urban_icl_prompt = """
Word: "meet ugly"
Example: "When Min Ho bumped into Kitty in the 1st episode, it was the meet ugly of the season, especially because they were enemies to almost lovers."
Meaning: The first encounter between individuals that is unpleasant or unfavorable, often leading to a complex relationship.

Word: "Slugging"
Example: "If your skin is dry after washing, you might add your hyaluronic acid after a rose water spritz, followed by your night creme, and finish with Vaseline as your 'slug' to trap the moisture."
Meaning: A skincare technique where a thick, occlusive layer (like Vaseline) is applied over other skincare products to lock in moisture, resulting in a dewy skin finish.

Word: "Fag"
Example: "See also faggot, twinky, and queer. Never gay. Unless said person is happy (the fag felt gay) which is the proper meaning of gay (gaeity, happy, cheerful)."
Meaning: A derogatory term used to describe a person, often related to their sexual orientation or behavior, and sometimes associated with judgments about their character or actions.

Word: {word}
Example: {example}
Meaning:
"""


urban_cot_prompt = """根据以下所有[例句]，分析词语[{word}]的含义，将其总结成一句简洁、通顺且易理解的定义。
注意：
    1. 用中文回答
    2. 你需要根据所有例句综合地一步一步地思考这个词的基本描述是什么，通常被用在什么语境或情况中，以及它通常会引发什么样的情感或反应。

==================
[例句]: 
{example}

请严格按照以下输出格式，避免输出多余的信息：
一步步分析:
结论:


"""

urban_causal_prompt = """
根据一系列带掩码（可能是网络俚语、流行词等）的例子，所有掩码都代表同一个词语，综合所有句子分析上下文并推断掩码单词（或短语）的含义。仔细思考这些例子，推断出可能的含义。如果不能从上下文中推断出含义，那么可以说“含义不清楚”。

要求：
1.按中文回答
2.不要一个句子一个句子单独分析，要综合所有句子的上下文语境

==================
带掩码单词的示例：
{example}
请严格按照以下输出格式，不要输出任何多余的信息：
分析:你的分析过程。
综合分析:综合所有分析，得到的结论，为一个综合你的分析的句子。
"""



urban_causal_mid_prompt = """
例子: 
{example}

任务:
随机实体替换:
   - 随机选择每个例句中非[MASKRD_PHRASE]的实体。
   - 对应生成一些新的、符合语境的实体。
   - 使用你生成的新的实体替换掉你原本选择的实体。
   - 确保语法和逻辑的连贯性。
   
任务示例：
原句子：
1.我真不喜欢吃妈妈做的[MASKRD_PHRASE]！
重构后的句子：
1.小明同学真不喜欢吃爸爸做的[MASKED_PHRASE]
该例子中将实体“我”、“妈妈”替换成了“小明同学”，“爸爸”。


请严格按照以下输出格式，不要输出任何多余的信息：
重构后的句子:
•句子一
•句子二
•句子三
......
"""

urban_causal_final_prompt = """指令：给定一个短语，使用提供的示例推断短语的含义。在观察用法示例时，要彻底解释上下文，以推断短语的微妙含义。将你的推理分解为循序渐进的逻辑，以达成全面的理解。
短语:{}
用法示例:
{}

1. 直接理解：这是直接根据短语在句子中的用法对短语的含义的解释。
   -可能的错误：对上下文或字面意思的误解。
   -定义：{}

2. 上下文推断：这是从短语被掩码的句子中推断出的意思。
   -可能的错误：由于缺乏上下文或者歧义而导致错误。
   -定义：{}

3. 重构上下文推断：这是当和短语相关的实体被替换，但同时实体与短语保持相同的关系的例句中推断出的含义。
   -可能的错误：新实体可能无法完全模仿原始上下文，导致理解偏差。
   -定义：{}

任务：综合考虑每个步骤的潜在错误和关键理解，对短语进行全面的定义或解释。

注意：
1. 用每个人，甚至是孩子都能理解的表达方式解释这个短语。如果需要，使用更多的单词来确保所有含义都清晰明了。用简单的单词清楚地解释这个短语。
2. 像字典一样解释，但确保清晰易读。

以下是一个给出定义的一个通用模板可以构建如下：
"[Word]表示[该词的基本描述]。 它经常被用于[该词的使用场景]。 这个短语[其他的细节，比如隐含意,情感倾向或者和该词有关的典型反应]。"
警告: 在生成定义时请严格遵守这个模板，以免生成不相关的信息。

生成示例:
短语: 白人饭
定义: 白人饭表示一种源自欧美，以生冷食材为主，无需过多加工，高纤维、高蛋白、低热量，且味道较为寡淡的简餐。它经常被用于表示对这种食物的抗拒。这个短语常带有一定的戏谑意味。

请严格按照以下json输出格式，不要输出任何多余的信息：
定义:
"""


urban_causal_legacy_propose_prompt = """Phrase: {word}

Example: {example}

Tasks:
1. Entity Candidate Generation:
   - For each entity in the example, including the Phrase, generate three alternative entities.
   - Include the original entity as a fourth candidate.

2. Sentence Reconstruction:
   - Replace "{word}" with "[MASKED_PHRASE]".
   - Replace each entity in the sentence with "[MASKED_ENTITY_1]", "[MASKED_ENTITY_2]", etc., in sequence.

Output:

1. Entity Replacement List:
   {{
     "[MASKED_PHRASE]": Candidate 1, Candidate 2, Candidate 3, Original Phrase;
     "[MASKED_ENTITY_1]": Candidate 1, Candidate 2, Candidate 3, Original Entity 1;
     ...
   }}

2. Reconstructed Sentence: (Sentence with "[MASKED_PHRASE]" and "[MASKED_ENTITY_X]")
"""

urban_causal_legacy_cot_prompt = """Instruction: Given a phrase and a reconstructed example sentence, use the reconstructed example to infer the meaning of the phrase. Additionally, consider the "Entity Candidates List". This list is crucial as it provides all possible entities or interpretations for each specific position in the original example sentence. By exploring these alternatives, you can gain a deeper understanding of the different contexts and nuances in which the phrase might be used. Methodically break down your reasoning into steps to arrive at a clear and well-explained conclusion.

Note:
1. Explain the phrase in simple words that everyone, including kids, can understand. Use more words if necessary to ensure clarity, and use easy words for explanations.
2. Write like a dictionary but in a very simple and easy-to-read manner.

Template for defining meanings, considering the variations of the phrase's use:
"[Word] refers to [basic description based on the reconstructed example]. In different contexts, it could mean [interpretations based on the entity candidates list]. It is often used [context or situation of usage based on the reconstructed example and entity candidates]. This expression [additional details like connotations, emotions, or typical reactions associated with the word]."

Warning: Please strictly follow this template for generating meanings to avoid irreversible consequences.

Phrase: {phrase}
Original Example: {example}
Reconstructed Example: {reconstructed_example}
Entity Candidates List: {entity_candidates_json}
   - This list contains alternative entities or interpretations for each entity in the original example. It offers insights into various possible meanings and uses of the phrase in different contexts.

Examples:
Conclusion: "[MASKED_PHRASE] refers to a skincare technique where an emollient or occlusive is applied to trap moisture. It's often used after washing the face to achieve a dewy look. This expression describes a specific beauty routine."

Conclusion: "[MASKED_PHRASE] refers to the practice of adopting measures to prevent COVID-19 infection. It's often used in the context of social and personal health safety. This expression signifies cautious behavior during the pandemic."

Conclusion: "[MASKED_PHRASE] refers to a period of facing winter's mental and physical challenges. It's often used to describe a time of perseverance and productivity. This expression implies resilience and determination."

Output:
A: Step 1: Analyze the reconstructed example and entity candidates.
   Step 2: Infer the various meanings and contexts for the phrase.
   Conclusion: ...
"""

model_knowledge_check = """
As part of an academic research project, I am evaluating phrases from Urban Dictionary to determine their uniqueness in relation to your training data as a large language model.

Phrase to Evaluate: {word}
Original Explanation: {meaning}
Original Usage Example: {example}

Honesty Reminder:
- You are reminded to be honest in your response. If you are not familiar with the phrase "{word}" and its specific meaning as provided, please respond with "I do not know" rather than guessing or making assumptions.

Evaluation Criteria:
- Indicate whether you are familiar with the phrase "{word}" and its specific meaning as outlined in the provided explanation.
- If you are already familiar with this phrase and its meaning, then the recommendation will be to discard it from the research dataset.
- If this phrase and its meaning are not known to you, or if they were not part of your training data, then the recommendation will be to retain it in the research dataset.

Ouptut format:

Decision: [Retain/Discard]
Explanation: [Brief explanation here]

Note: Replace [Retain/Discard] and [Brief explanation here] with your specific decision and explanation respectively, based on your familiarity with the phrase.
"""


clean_data = """
Given the phrase (or word) "{word}" from Urban Dictionary, along with its usage example "{example}" and meaning "{meaning}", evaluate the content for its suitability in our dataset by following these steps:

1. Examine for NSFW Content:
   - Check for Explicit Language in Example: Does the usage example contain sexually explicit language or vivid adult content? (Yes/No)
   - Assess for Violence in Example: Are there descriptions or glorifications of violence in the usage example? (Yes/No)

2. Assess Aggressiveness in Meaning:
   - Check for Extreme Hate Speech in Meaning: Does the definition include severe forms of hate speech or incitement to violence against specific groups? (Yes/No)

3. Final Evaluation:
   - Retention Criteria: The content should be retained unless any of the answers to the above questions is "Yes".
   - Discard Criteria: The content should be discarded if any answer to the above questions is "Yes".
   - Final Decision: Should the content be retained or discarded? (Retain/Discard)

4. Decision Justification:
   - Provide a brief explanation for your decision. This explanation will help understand the context and rationale behind your decision.

Ouptut format:

Decision: [Retain/Discard]
Explanation: [Brief explanation here]

Note: Replace [Retain/Discard] and [Brief explanation here] with your specific decision and explanation respectively.
"""

transform_dict_data = """Change internet slang or memes into simple dictionary entries. Make sure they're easy to understand:

Input: 
- Phrase: {word}
- Original Explanation: {meaning}
- Original Example: {example}

Process:
1. Explain the phrase in simple words that everyone, even kids, can understand. If needed, use more words to make sure all meanings are clear.
2. Provide a concise example that demonstrates the usage of the phrase. The example should subtly convey the true meaning of the phrase through its semantic context, emotions, and cultural background, allowing the meaning to be inferred without explicitly explaining it (DO NOT explain it directly in usage example!!!).
3. Keep the real meaning of the phrase as people know it online.
4. Write like a dictionary but very simple and easy to read.

Explanation Transformation Guidelines:
A simplified and general template for defining the meanings of internet slang or memes can be structured as follows:
"[Word] refers to [basic description of the word]. It is often used [context or situation of usage]. This expression [additional details like connotations, emotions, or typical reactions associated with the word]."
Warning: Please strictly follow this template for conversion to avoid irreversible consequences.

Explanation Transformation Example:

Phrase: Slugging
Original Explanation: "Involves coating the skin in some kind of emollient or occlusive as a way of trapping moisture, for a dewy, slimy look."
Transformed Explanation: "Slugging refers to a skincare technique where an emollient or occlusive is applied to trap moisture. It's often used after washing the face to achieve a dewy look. This expression describes a specific beauty routine."

Phrase: Coviding
Original Explanation: "Taking precautions to prevent the infection and spread of COVID-19."
Transformed Explanation: "Coviding refers to the practice of adopting measures to prevent COVID-19 infection. It's often used in the context of social and personal health safety. This expression signifies cautious behavior during the pandemic."

Phrase: The Winter Arc
Original Explanation: "A time to face challenges of winter and get things done."
Transformed Explanation: "The Winter Arc refers to a period of facing winter's mental and physical challenges. It's often used to describe a time of perseverance and productivity. This expression implies resilience and determination."      

Example Transformation Guidelines:
1. Clear Example: Create an example that uses the phrase in a way that's easy to understand.
2. True to Online Use: Make sure the example match how the phrase is used online, and the example should conform to the cultural background and true meaning (both original and transformed explaination) of this phrase.
3. Ensure that the entire phrase appears in transformed example completely.

Output:
- Analysis: [Talk about how you made the explanation and example clear, true to online culture, and easy to understand]
- Transformed Explanation: [Simple and clear explanation of the phrase]
- Transformed Usage Example: [Simple example that shows how the phrase is used, helping to understand its meaning]

Note: Fill in the placeholders with real internet slang or meme information. Use language that is easy, common, and straightforward in both the explanation and example.]"""

generate_new_meaning = """Input:

Phrase: {word}
Explanation: {transformed_meaning}

Task: 

I've provided you with an explanation of the phrase in the dictionary. Rewrite the explanation to preserve its original intent and meaning. Your goal is to maintain a similar sentence length and structure but use different wording. Focus on substituting words and phrases with their synonyms while ensuring the meaning remains clear and consistent with the original explanation. Avoid altering the fundamental concepts and essential terms.
A simplified and general template for defining the meanings of internet slang or memes can be structured as follows:
"[Word] refers to [basic description of the word]. It is often used [context or situation of usage]. This expression [additional details like connotations, emotions, or typical reactions associated with the word]."
Warning: Please strictly follow this template for conversion (NEVER modify words that already appear in the template!) to avoid irreversible consequences.

Phrase: Fusk
Explanation: Fusk refers to a word used to show that one is really annoyed. It is often used among friends or group members who understand and support each other. This expression shows strong feelings without using bad language.\"",
            
New Transformed Explanation 1: Fusk refers to a term signaling that someone is extremely irritated. It is often used among companions or team members who share mutual understanding and camaraderie. This phrase conveys intense emotion without resorting to offensive language.
New Transformed Explanation 2: Fusk refers to an expression indicating significant annoyance. It is often used within circles of acquaintances or allies familiar with each other's sentiments. This expression communicates deep emotions while avoiding vulgar terms.
New Transformed Explanation 3: Fusk refers to a slang that denotes a high level of vexation. It is often used among peers or in-group contexts where there's a strong sense of fellowship. This term expresses powerful feelings without the use of profanity.
New Transformed Explanation 4: Fusk refers to a colloquialism used to express considerable exasperation. It is often used by a cluster of friends or group affiliates who are supportive and empathetic towards one another. This locution articulates vehement emotions without the employment of crude words.
New Transformed Explanation 5: Fusk refers to a word used to show that one is really annoyed. It is often used among friends or group members who understand and support each other. This expression shows strong feelings without using bad language.

Output:

New Transformed Explanation 1: [Your newly crafted explanation]
New Transformed Explanation 2: [Your newly crafted explanation]
New Transformed Explanation 3: [Your newly crafted explanation]
New Transformed Explanation 4: [Your newly crafted explanation]


Note: Replace the placeholders in brackets [ ] with your content. New Transformed Explanation should start with {word}.
"""