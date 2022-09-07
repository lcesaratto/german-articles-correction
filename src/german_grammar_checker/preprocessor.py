import re


class Preprocessor:
    """
    Preprocessing tweet contents
    """
    def __init__(self, pipeline=None):
        """Init with custom pipeline or default all preprocessing steps.
        """
        if pipeline is None:
            pipeline = [*self.mapping]
        self.pipeline = pipeline

    def __call__(self, text: str) -> str:
        """Apply defined pipeline.
        """
        for f_name in self.pipeline:
            text = self.mapping[f_name](text)

        return text

    @property
    def mapping(self):
        """Mapping to method names
        """
        return {
            'hyperlinks': self.hyperlinks,
            'mentions': self.mentions,
            'hashtags': self.hashtags,
            'retweet': self.retweet,
            'emojis': self.emojis,
            'smileys': self.smileys,
            'repetitions': self.repetitions,
            'spaces': self.spaces,
        }

    @staticmethod
    def hyperlinks(text: str) -> str:
        """Removes hyperlinks, replaces with token url
        """
        return re.sub(r'\S*https?:\S*', ' url ', text)

    @staticmethod
    def mentions(text: str) -> str:
        """Removes mentions, replaces with token mention
        """
        return re.sub(r'@\w*', ' mention ', text)

    @staticmethod
    def hashtags(text: str) -> str:
        """Removes hashtags, replaces with hashtag content
        """
        return re.sub(r'#(\S+)', r' \1 ', text)

    @staticmethod
    def retweet(text: str) -> str:
        """Removes retweets
        """
        text = re.sub(r'\brt\b', ' ', text)
        return re.sub(r'\bRT\b', ' ', text)

    @staticmethod
    def repetitions(text: str) -> str:
        """Convert more than 2 letter repetitions to 2 letter
        """
        return re.sub(r'(.)\1+', r'\1\1', text)

    @staticmethod
    def spaces(text: str) -> str:
        """Normalize spaces
        """
        return re.sub(r'\s+', ' ', text)

    @staticmethod
    def emojis(text: str) -> str:
        """Removes emojis, replaces with token emoji
        src: https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
        """
        pattern = re.compile("["
                             u"\U0001F600-\U0001F64F"  # emoticons
                             u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                             u"\U0001F680-\U0001F6FF"  # transport & map symbols
                             u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                             "]+", flags=re.UNICODE)

        return pattern.sub(r' emoji ', text)

    @staticmethod
    def smileys(text: str) -> str:
        """Removes smileys, replace with token smiley
        src: https://github.com/abdulfatir/twitter-sentiment-analysis/blob/master/code/preprocess.py
        """
        # Smile -- :), : ), :-), (:, ( :, (-:, :')
        text = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' smiley ', text)
        # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
        text = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' smiley ', text)
        # Love -- <3, :*
        text = re.sub(r'(<3|:\*)', ' smiley ', text)
        # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
        text = re.sub(r'(;-?\)|;-?D|\(-?;)', ' smiley ', text)
        # Sad -- :-(, : (, :(, ):, )-:
        text = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' smiley ', text)
        # Cry -- :,(, :'(, :"(
        text = re.sub(r'(:,\(|:\'\(|:"\()', ' smiley ', text)

        return text