# import modules
import argparse
import json
import os
import random
from itertools import islice

import nltk
import pandas as pd
import requests
import wikipedia
import wptools
from SPARQLWrapper import SPARQLWrapper, JSON


class Extractor:
    def __init__(self, verbose=False):
        self.sentence_tokenize = nltk.sent_tokenize
        self.keywords = ['architect', 'mathematician', 'painter', 'politician', 'singer', 'writer']
        agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
        self.endpoint = SPARQLWrapper('https://query.wikidata.org/sparql', agent=agent)
        self.endpoint.setReturnFormat(JSON)
        self.verbose = verbose
        self.A = ['singer', 'writer', 'painter']
        self.Z = ['architect', 'politician', 'mathematician']

    def extract(self, n_sentences: int, n_people: int):
        if self.verbose:
            print('▶ extracting the ids')
        if os.path.isfile('keywords_ids.json'):
            with open('keywords_ids.json') as kfile:
                ids = json.loads(kfile.read())
        else:
            ids = {}
            for keyword in self.keywords:
                if self.verbose:
                    print('▶ keyword:', keyword)
                ids[keyword] = self.get_ids(keyword)
            with open('keywords_ids.json', 'w') as out:
                json.dump(ids, out)

        data = {}
        if self.verbose:
            print('▶ parsing the descriptions')
        for keyw, ppl_ids in ids.items():
            if self.verbose:
                print('▶', keyw)
            data[keyw] = []
            random.shuffle(ppl_ids)
            if self.verbose:
                print(f'▶ number of {keyw}s is {len(ppl_ids)}')
            counter = 0
            for id_ in ppl_ids:
                if counter == n_people:
                    break
                if self.verbose:
                    print('▶ counter', counter)
                if self.verbose:
                    print('▶ id_', id_)
                article = {}
                try:
                    t, d = self.get_title_and_description(id_)
                    if not isinstance(d, str):
                        if self.verbose:
                            print('▶ FAILED - Description is empty')
                        continue
                except LookupError:
                    if self.verbose:
                        print('▶ FAILED - LookupError')
                    continue

                try:
                    c = self.get_content(wikipedia.page(t), n_sentences=n_sentences)
                    c = self.sentence_tokenize(c)[:10]
                    if len(c) < n_sentences:
                        if self.verbose:
                            print('▶ FAILED - number of sentences is not sufficient')
                        continue
                    c = " ".join(c)
                except (wikipedia.PageError, wikipedia.DisambiguationError):
                    if self.verbose:
                        print('▶ FAILED - PageError / DisambiguationError')
                    continue
                counter += 1
                article['title'] = t
                article['description'] = d
                article['content'] = c
                data[keyw].append(article)

        if self.verbose:
            print('▶ storing the data into csv')
        self.get_csv(data)
        return data

    def get_ids(self, keyword: str):
        result = requests.get('https://www.wikidata.org/w/api.php',
                              params={'format': 'json',
                                      'action': 'wbsearchentities',
                                      'search': keyword,
                                      'language': 'en'})
        result = result.json()
        key_id = result['search'][0]['id']
        query = '''SELECT ?personLabel
        WHERE
        {
        ?person wdt:P106 wd:''' + key_id + '''.
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE]". }
        }
        '''

        self.endpoint.setQuery(query)
        results = self.endpoint.query()

        uri = json.loads(''.join(map(lambda x: x.decode(), results)))

        ids = []
        for person in uri['results']['bindings']:
            ids.append(person['personLabel']['value'])

        return ids

    @staticmethod
    def get_title_and_description(id_: str):
        page = wptools.page(wikibase=id_, silent=True)
        page.get_wikidata()
        title = page.data['title']
        description = page.data['description']

        return title, description

    def get_content(self, title: str, n_sentences: int):
        page = wikipedia.page(title)
        content = page.content
        sentences = [sent.strip() for sent in islice(self.sentence_tokenize(content), n_sentences)]

        return ' '.join(sentences)

    def get_csv(self, data):
        category = []
        datatype = []
        title = []
        description = []
        content = []
        for cat, persons in data.items():
            for person in persons:
                category.append(cat)
                if cat in self.A:
                    datatype.append('A')
                else:
                    datatype.append('Z')
                title.append(person['title'])
                description.append(person['description'])
                content.append(person['content'])
        df = pd.DataFrame({'title': title, 'description': description, 'category': category,
                           'datatype': datatype, 'text': content})
        df.to_csv('data.csv', index=False)


def main(n_sentences, n_people, verbose):
    extractor = Extractor(verbose)
    extractor.extract(n_sentences, n_people)


# main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corpus Extractor")
    parser.add_argument("--n_people", type=int, default=30, help="number of persons to extract for each domain")
    parser.add_argument("--n_sentences", type=int, default=10, help="number of sentences to extract for each person")
    parser.add_argument('--verbose', help='print out the logs (default: False)', action='store_true')
    args = parser.parse_args()
    main(args.n_sentences, args.n_people, args.verbose)
