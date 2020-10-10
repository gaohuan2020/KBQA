from elasticsearch import Elasticsearch

es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
es.close()
e1 = {
    "first_name": "nitin",
    "last_name": "panwar",
    "age": 27,
    "about": "Love to play cricket",
    "interests": ['sports', 'music'],
}

e2 = {
    "first_name": "Jane",
    "last_name": "Smith",
    "age": 32,
    "about": "I like to collect rock albums",
    "interests": ["music"]
}

e3 = {
    "first_name": "Douglas",
    "last_name": "Fir",
    "age": 35,
    "about": "I like to build cabinets",
    "interests": ["forestry"]
}
res = es.delete(index='megacorp', id=1)
res = es.delete(index='megacorp', id=2)
res = es.delete(index='megacorp', id=3)
res = es.index(index='megacorp', id=1, body=e1)
res = es.index(index='megacorp', id=2, body=e2)
res = es.index(index='megacorp', id=3, body=e3)
res = es.get(index='megacorp', id=3)
res = es.search(index='megacorp',
                body={'query': {
                    'match': {
                        'first_name': 'nitin0'
                    }
                }})
print(res['hits']['hits'])
res = es.search(
    index='megacorp',
    body={'query': {
        'bool': {
            'must': [{
                'match': {
                    'first_name': 'Douglas'
                }
            }]
        }
    }})
print(res['hits']['hits'])