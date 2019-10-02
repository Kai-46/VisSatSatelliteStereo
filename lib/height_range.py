
# ===============================================================================================================
#  This file is part of Creation of Operationally Realistic 3D Environment (CORE3D).                            =
#  Copyright 2019 Cornell University - All Rights Reserved                                                      =
#  -                                                                                                            =
#  NOTICE: All information contained herein is, and remains the property of General Electric Company            =
#  and its suppliers, if any. The intellectual and technical concepts contained herein are proprietary          =
#  to General Electric Company and its suppliers and may be covered by U.S. and Foreign Patents, patents        =
#  in process, and are protected by trade secret or copyright law. Dissemination of this information or         =
#  reproduction of this material is strictly forbidden unless prior written permission is obtained              =
#  from General Electric Company.                                                                               =
#  -                                                                                                            =
#  The research is based upon work supported by the Office of the Director of National Intelligence (ODNI),     =
#  Intelligence Advanced Research Projects Activity (IARPA), via DOI/IBC Contract Number D17PC00287.            =
#  The U.S. Government is authorized to reproduce and distribute copies of this work for Governmental purposes. =
# ===============================================================================================================

def height_range(rpc_models):
    z_min_list = []
    z_max_list = []
    for i in range(len(rpc_models)):
        assert (rpc_models[i].altScale > 0)

        tmp = rpc_models[i].altOff - 0. * rpc_models[i].altScale
        z_min_list.append(tmp)

        tmp = rpc_models[i].altOff + 0.7 * rpc_models[i].altScale
        z_max_list.append(tmp)
    # take intersection
    z_min = max(z_min_list)
    z_max = min(z_max_list)

    # manually set for mvs3dm dataset
    # z_min = 0
    # z_max = 100

    # z_min = 0
    # z_max = 100

    return z_min, z_max
