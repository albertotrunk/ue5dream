def clean_prompt(prompt):
  badchars = re.compile(r'[/\\]')
  prompt = badchars.sub('_', prompt)
  if len(prompt) > 100:
    prompt = f'{prompt[:100]}…'
  return prompt

def format_filename(timestamp, seed, index, prompt):
  string = save_location
  string = string.replace('%T', f'{timestamp}')
  string = string.replace('%S', f'{seed}')
  string = string.replace('%I', f'{index:02}')
  string = string.replace('%P', clean_prompt(prompt))
  return string

def save_image(image, **kwargs):
  filename = format_filename(**kwargs)
  os.makedirs(os.path.dirname(filename), exist_ok=True)
  image.save(filename)
