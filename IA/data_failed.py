class Data_Failed:

    def __init__(self, img_path, datetime, start_time, rpm, diameter):
        self.img_path = img_path
        self.datetime = datetime
        self.start_time = start_time
        self.rpm = rpm 
        self.diameter = diameter 


    def save_data(self):
        min_faield = self.datetime - self.start_time
        turn = self.rpm * min_faield
        circle = 3,14 * self.diameter
        meter = turn * circle
        # vetor = []
        # vetor.append(meter)
        return meter

        