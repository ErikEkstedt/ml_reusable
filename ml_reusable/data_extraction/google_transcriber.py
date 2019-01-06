import os
import json
from os.path import join

from google.cloud import speech
from google.cloud.speech import enums, types
from google.oauth2.service_account import Credentials


class GoogleTranscriber(object):
    def __init__(self, credential_path, sample_rate=16000, lang='swe'): 
        self.credential_path = credential_path
        self.credentials = self._credentials()
        self.sample_rate = sample_rate
        if lang.lower() == 'swe':
            self.language = 'sv-SE'
        else:
            self.language = 'en-US'
        self.config = self._config()
        self.client = self._client()

    def _credentials(self):
        return Credentials.from_service_account_file(self.credential_path)

    def _config(self):
        return types.RecognitionConfig(
                encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
                language_code=self.language,
                enable_word_time_offsets=True)

    def _client(self):
        return speech.SpeechClient(credentials=self.credentials)

    def __call__(self, path):
        return self.transcribe(path)

    def transcribe(self, path):
        ''' Argument: path to wav-file to transcribe '''
        with open(path, 'rb') as audio_file:
            content = audio_file.read()
        audio = types.RecognitionAudio(content=content)
        response = self.client.recognize(self.config, audio)
        return self.reformat_response(response)

    def reformat_response(self, response):
        transcripts = []
        for result in response.results:
            for alternative in result.alternatives:
                transcript = {}
                transcript['utterence'] = alternative.transcript
                transcript['words'] = []
                for word_info in alternative.words:
                    word = word_info.word
                    start_time = word_info.start_time
                    end_time = word_info.end_time
                    transcript['words'].append(
                            {
                                'w': word,
                                't0': start_time.seconds + start_time.nanos * 1e-9,
                                't1': end_time.seconds + end_time.nanos * 1e-9
                                }
                            )
                transcripts.append(transcript)
        self.response = response  # store entire response from google
        return transcripts

    def save_transcript(self, filepath, transcripts, verbose=True):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(transcripts, f)
        if verbose: print(f'Saved transcript to: {filepath}')


def transcribe_and_save(source, target, credential_path,
        sample_rate=16000, lang='swe'):
    transcriber = GoogleTranscriber(
            credential_path=credential_path,
            sample_rate=sample_rate, 
            lang=lang)
    transcripts = transcriber(source)
    transcriber.save_transcript(target, transcripts, verbose=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='GoogleTranscriber')
    parser.add_argument('-c', '--credential', type=str,
            help='Path to google credentials (json) ')
    parser.add_argument('-s', '--source', type=str,
            help='Path to wav-file to be transcribed')
    parser.add_argument('-t', '--target', type=str,
            help='Path to output json-file')
    parser.add_argument('-l', '--language', type=str, default='swe',
            help='language (default: swe) else english')
    parser.add_argument('--sample_rate', type=int, default=16000,
            help='sample rate of audio (default: 16000)')
    args = parser.parse_args()
    
    transcriber = GoogleTranscriber(
            credentials=args.credential,
            sample_rate=args.sample_rate, 
            lang=args.language)
    transcripts = transcriber(args.source)
    transcriber.save_transcript(args.target, transcripts)
