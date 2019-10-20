import json
import requests


class BingSpellCheck(object):

    ENDPOINT = 'https://api.cognitive.microsoft.com/bing/v7.0/SpellCheck'

    def __init__(self, key):
        """ Constructor.

        :param str key:
        :return void:
        """
        self.api_key = key

    def spell_check(self, text, mode='spell'):
        """ Correct spelling in a string.

        :param str text:
        :param str mode:
        :return str:
        """
        flagged_tokens = self._get_flagged_tokens(text, mode)
        return self._replace_flagged_tokens(text, flagged_tokens)

    def _get_flagged_tokens(self, text, mode):
        """ Make the request to Bing to get spell corrections for the given string.

        :param str text:
        :param str mode:
        :return dict:
        """
        data = {'text': text}

        params = {
            'mkt': 'en-us',
            'mode': mode
        }

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Ocp-Apim-Subscription-Key': self.api_key,
        }

        response = requests.post(
            self.ENDPOINT,
            headers=headers,
            params=params,
            data=data
        )

        return response.json()['flaggedTokens']

    def _replace_flagged_tokens(text, flagged_tokens):
        """ Iterate through flagged tokens and replace them with the suggested tokens.

        :param str text:
        :param list flagged_tokens:
        :return str:
        """
        for flagged_ in flagged_tokens:
            text = text.replace(flagged['token'], flagged['suggestions'][0]['suggestion'])

        return text
