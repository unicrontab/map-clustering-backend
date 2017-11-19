import json

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import pairwise_distances_argmin
    import numpy as np
except:
    import unzip_requirements
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import pairwise_distances_argmin
    import numpy as np

def main(event, context):
    body = {}
    dataObject = []
    if 'body' in event.keys():
        body = json.loads(event['body'])
        dataObject = body['data']
    else:
        return createResponse({ 'error': 'no data'})
    
    X = np.empty(shape=[0,2])
    print(X)
    for address in dataObject:
        lat = float(address['location']['lat'])
        lng = float(address['location']['lng'])
        X = np.append(X, [[lat, lng]], axis=0)

    n_clusters = 2
    k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    k_means.fit(X)
    k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
    k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

    prediction = k_means.predict(X)
    print(prediction)

    clusterData = []
    for index, address in enumerate(dataObject):
        lat = float(address['location']['lat'])
        lng = float(address['location']['lng'])
        cluster = int(prediction[index])
        clusterData.append({
            'address': address['address'],
            'lat': lat,
            'lng': lng,
            'cluster': cluster,
        })
    
    dataWithMetaData = {
        'clusters': n_clusters,
        'algo': 'k_means',
        'addresses': len(dataObject),
        'clusterData': clusterData
    }
    return createResponse(dataWithMetaData)

def createResponse(modelOutput):
    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Credentials': 'true'
        },
        'body': json.dumps(modelOutput)
    }


if __name__ == '__main__':
    data = [
        {
            'address': '1 to -2',
            'location': {
                'lat': 1,
                'lng': -2,
            }
        },
        {
            'address': '2 to -2',
            'location': {
                'lat': 2,
                'lng': -2,
            }
        },
        {
            'address': '3 to -2',
            'location': {
                'lat': 3,
                'lng': -2,
            }
        },
        {
            'address': '10 to 3',
            'location': {
                'lat': 10,
                'lng': 3,
            }
        },
        {
            'address': '11 to 5',
            'location': {
                'lat': 11,
                'lng': 5,
            }
        },
        {
            'address': '9 to 4',
            'location': {
                'lat': 9,
                'lng': 4,
            }
        }
    ]
    body = {
        'algo': 'predict',
        'data': json.dumps(data)
    }
    testEvent = {}
    testEvent['body'] = json.dumps(body);
    print(json.dumps(testEvent, indent=4))
    # print(json.dumps(main(testEvent, ''), indent=4))
