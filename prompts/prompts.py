# -*- coding: utf-8 -*-
####------LaMP_1------###
def create_classification_citation_prompt(inp, profile, max_length, tokenizer):
    prompts = []
    per_p_max_length = (max_length - 2 * (len(profile) - 1)) // len(profile)
    saved_tokens = 0

    for p in profile:
        tokens = tokenizer(p["title"], max_length=per_p_max_length + saved_tokens - 2, truncation=True)
        saved_tokens += per_p_max_length - len(tokens['input_ids']) - 2
        new_title = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
        prompts.append(f'"{new_title}"')


    profile_str = ", ".join(prompts)
    
    phrase = "who has written"
    idx = inp.find(phrase)
    
    if idx == -1:
        instruction = inp
    
    instruction = inp[:idx] + f"who has previously written paper titles {profile_str}, and now has "+ inp[idx + len(phrase):]
    return instruction

####------LaMP_1------###
def create_classification_news_prompt(inp, profile, max_length, tokenizer): # good
    per_p_max_length = (max_length - 1 - 2 * (len(profile) - 1)) // len(profile)
    saved_tokens = 0
    prompts = []
    for p in profile:
        needed_part_len = len(tokenizer(f'the category for the article: " " is "{p["category"]}" ')['input_ids'])
        tokens = tokenizer(p["text"], max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
        saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
        new_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
        prompt = f'the category for the article: "{new_text}" is "{p["category"]}" '
        prompts.append(prompt)
    return f'{", and ".join(prompts)}. {inp}'

####------LaMP_2------###
def create_classification_movies_prompt(inp, profile, max_length, tokenizer):
    prompts = []
    per_p_max_length = (max_length - 2 * (len(profile) - 1)) // len(profile)
    saved_tokens = 0

    for p in profile:
        # tokenize each movie description
        tokens = tokenizer(p["description"], max_length=per_p_max_length + saved_tokens - 2, truncation=True)
        saved_tokens += per_p_max_length - len(tokens['input_ids']) - 2
        new_desc = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
        prompts.append(f'the tag for the movie: "{new_desc}" is "{p["tag"]}"')


    profile_str = ", and ".join(prompts)

    instruction = f'Given the user previous movie tag pairs: {profile_str}. {inp}'
    
    instruction = instruction.replace("] description", "]. description")
    
    return instruction

####------LaMP_3------###
def create_classification_review_prompt(inp, profile, max_length, tokenizer):
    prompts = []
    per_p_max_length = (max_length - 2 * (len(profile) - 1)) // len(profile)
    saved_tokens = 0

    for p in profile:
        needed_part_len = len(tokenizer(f'{p["score"]} is the score for " " ')['input_ids'])
        tokens = tokenizer(
            p["text"],
            max_length=per_p_max_length + saved_tokens - needed_part_len,
            truncation=True
        )
        saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
        new_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]

        prompts.append(f'{p["score"]} is the score for "{new_text}"')

    profile_str = ", ".join(prompts)

    instruction = f'Given the user previous review-score pairs: {profile_str}. {inp}'

    return instruction

####------LaMP_4------###
def create_generation_news_prompt(inp, profile, max_length, tokenizer):
    prompts = []
    per_p_max_length = (max_length - 2 * (len(profile) - 1)) // len(profile)
    saved_tokens = 0

    for p in profile:
        needed_part_len = len(tokenizer(f'"{p["title"]}" is the title for " " ')['input_ids'])
        tokens = tokenizer(
            p["text"],
            max_length=per_p_max_length + saved_tokens - needed_part_len,
            truncation=True
        )
        saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
        new_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]

        prompts.append(f'"{p["title"]}" is the title for "{new_text}"')

    profile_str = ", ".join(prompts)

    instruction = f'Given the user’s previous article-headline pairs: {profile_str}. {inp}'

    return instruction

####------LaMP_5------###
def create_generation_paper_prompt(inp, profile, max_length, tokenizer):
    prompts = []
    per_p_max_length = (max_length - 2 * (len(profile) - 1)) // len(profile)
    saved_tokens = 0

    for p in profile:
        # calculate available length for each abstract snippet
        needed_part_len = len(tokenizer(f'"{p["title"]}" is a title for " " ')['input_ids'])
        tokens = tokenizer(p["abstract"], max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
        saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
        new_abstract = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]

        prompts.append(f'"{p["title"]}" is a title for "{new_abstract}"')

    profile_str = ", ".join(prompts)

    instruction = f'Given the user’s previous abstract-title pairs: {profile_str}. {inp}'

    return instruction

####------LaMP_7------###
def create_parphrase_tweet_prompt(inp, profile, max_length, tokenizer):
    prompts = []
    per_p_max_length = (max_length - 2 * (len(profile) - 1)) // len(profile)
    saved_tokens = 0

    for p in profile:
        # calculate available length for each tweet snippet
        needed_part_len = len(tokenizer(f'"" ')['input_ids'])
        tokens = tokenizer(
            p["text"],
            max_length=per_p_max_length + saved_tokens - needed_part_len,
            truncation=True
        )
        saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
        new_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]

        # build one training example
        prompts.append(f'"{new_text}"')

    # join multiple examples naturally
    profile_str = ", ".join(prompts)
    
    inp = inp.replace(
            "Paraphrase the following tweet without any explanation before or after it:",
            "Paraphrase the following tweet without any explanation before or after it following the user's tweeting patterns. Tweet:"
            )
    
    # final instruction
    instruction = f'Given the user’s previous tweets: {profile_str}. {inp}'
    return instruction

def create_prompt_generator(num_retrieve, max_length = 512, tokenizer = None):

    def prompt(inp, profile, task, mode):
        selected_profs = profile[:num_retrieve]
        factor = 0.6
        while True:
            try:
                max_len_prompt = max_length - min(len(tokenizer(inp)['input_ids']), int(factor * max_length))
                if task == "LaMP_1":
                    return create_classification_citation_prompt(inp, selected_profs, max_len_prompt, tokenizer)
                elif task == "LaMP_2":
                    return create_classification_movies_prompt(inp, selected_profs, max_len_prompt, tokenizer)
                elif task == "LaMP_3":
                    return create_classification_review_prompt(inp, selected_profs, max_len_prompt, tokenizer)
                elif task == "LaMP_4":
                    return create_generation_news_prompt(inp, selected_profs, max_len_prompt, tokenizer)
                elif task == "LaMP_5":
                    return create_generation_paper_prompt(inp, selected_profs, max_len_prompt, tokenizer)
                elif task == "LaMP_7":
                    return create_parphrase_tweet_prompt(inp, selected_profs, max_len_prompt, tokenizer)
            except:
                factor -= 0.1
                if factor < 0:
                    print("not possible")
                    return inp
    return prompt