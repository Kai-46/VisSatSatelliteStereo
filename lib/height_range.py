
def height_range(rpc_models):
    z_min_list = []
    z_max_list = []
    for i in range(len(rpc_models)):
        assert (rpc_models[i].altScale > 0)

        tmp = rpc_models[i].altOff - 0.* rpc_models[i].altScale
        z_min_list.append(tmp)

        tmp = rpc_models[i].altOff + 0.7 * rpc_models[i].altScale
        z_max_list.append(tmp)
    # take intersection
    z_min = max(z_min_list)
    z_max = min(z_max_list)

    # manually set for mvs3dm dataset
    z_min = 10
    z_max = 80

    return z_min, z_max