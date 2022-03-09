from collections import Counter

def fixate_attention(predictions, attentional_limit):
    n_perceptions = len(predictions["U"])
    inverted_predictions = [{}] * n_perceptions

    for direction, perception_action in predictions.items():
        for perception, action in perception_action.items():
            temp = inverted_predictions[perception].copy()
            temp[direction] = action["prediction"]
            inverted_predictions[perception] = temp

    higher_predictions = []
    for perception, direction_prediction in enumerate(inverted_predictions):
        predictions_without_position = {}
        for direction, prediction in direction_prediction.items():
            predictions_without_position[direction + str(perception)] = prediction
        action_counter = Counter(predictions_without_position)
        most_common_predictions = action_counter.most_common()[0:attentional_limit]
        for prediction in most_common_predictions:
            higher_predictions.append(prediction)

    fixated_predictions = {}
    for prediction in higher_predictions:
        direction = prediction[0][0]
        perception = int(prediction[0][1:])
        fixated_predictions[direction+str(perception)] = predictions[direction][perception]

    return fixated_predictions
