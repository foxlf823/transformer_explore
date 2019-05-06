

class Alphabet:
    def __init__(self, name, label=False):
        self.name = name
        self.UNKNOWN = "</unk>"
        self.PAD = "</pad>"
        self.label = label
        self.instance2index = {}
        self.instances = []
        self.next_index = 0

        if not self.label:
            self.add(self.PAD)
            self.add(self.UNKNOWN)
        
    def add(self, instance):
        if instance not in self.instance2index:
            self.instances.append(instance)
            self.instance2index[instance] = self.next_index
            self.next_index += 1

    def size(self):
        return self.next_index

    def get_index(self, instance):
        if not self.label:
            if instance in self.instance2index:
                return self.instance2index[instance]
            else:
                return self.instance2index[self.UNKNOWN]
        else:
            if instance in self.instance2index:
                return self.instance2index[instance]
            else:
                raise RuntimeError("{} not exist".format(instance))

    def get_instance(self, index):
        if index >= 0 and index < self.next_index:
            return self.instances[index]
        else:
            raise RuntimeError("{} not exist".format(index))

    def iteritems(self):
        return self.instance2index.items()
