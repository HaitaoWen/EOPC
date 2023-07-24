from tool import *

from .replay.er import ER
from .replay.icarl import iCaRL
from scheme.replay.podnet.podnet import PODNet
from .structure.aanet.aanet import AANet
from .replay.opc.opc import OPC


def init_scheme(model, scenario):
    start = 1
    Scheme = eval(args.scheme)
    if args.resume:
        memory.load_memory()
        start = eval(args.resume[args.resume.rfind('task') + 4: args.resume.rfind('.')]) + 1
        for taskid, traindata in enumerate(scenario, start=1):
            if taskid == start:
                break
            scheme = Scheme(model, traindata, taskid)
            model = scheme.model
        model = load_model(model)
    return Scheme, model, start
