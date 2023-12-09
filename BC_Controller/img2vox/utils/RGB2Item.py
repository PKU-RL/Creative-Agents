import numpy as np



forbidden_items = [
    2,
    3,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    19,
    21,
    23,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    37,
    38,
    39,
    40,
    44,
    46,
    47,
    51,
    52,
    54,
    55,
    56,
    58,
    59,
    60,
    61,
    62,
    63,
    65,
    66,
    68,
    69,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    81,
    83,
    84,
    85,
    86,
    87,
    88,
    92,

    93,
    94,
    96,
    97,
    100,
    104,
    101,
    102,
    104,
    105,
    106,
    107,
    110,
    111,
    113,
    115,
    116,
    117,
    118,
    119,
    120,
    122,
    123,
    124,
    126,
    127,
    129,
    130,
    131,
    132,
    137,
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
    151,
    152,
    153,
    154,
    157,
    158,
    166,
    170,
    171,

    175,
    176,
    177,
    178,
    182,
    183,
    184,
    185,
    186,
    187,
    188,
    189,
    190,
    191,
    192,
    193,
    194,
    195,
    196,
    197,
    198,
    207,
    217
]


non_place_items=[
    6,
    12,
    13,
    26,
    27,
    28,
    31,
    32,
    34,
    37,
    38,
    39,
    40,
    50,
    51,
    55,
    59,
    63,
    64,
    65,
    66,
    68,
    69,
    71,
    75,
    76,
    77,
    81,
    83,
    92,
    93,
    94,
    104,
    105,
    106,
    111,
    115,
    116,
    119,
    122,
    132,
    141,
    142,
    143,
    145,
    157,
    166,
    171,
    175,
    178,
    193,
    194,
    195,
    196,
    197,
    198,
    199,
    200,
    205,
    207,
    208,
    209,
    210,
    211,
    213,


    217,
    218,
    235,
    236,
    237,
    238,
    239,
    242,
    243,
    244,
    245,
    246,
    247,
    248,
    249,
    250,

    79,
    174,
    212,
    147,
    148,


]



def DeltaRGB(R,G,B,o2r):
    return o2r - np.array([R,G,B])

def RGB2Item(R,G,B):
    ord2rgb = np.load("/home/ps/Desktop/DDPM/MC_image/Pix2Vox_Dataset/icons/ord2rgb.npy")
    DRGB = DeltaRGB(R,G,B,ord2rgb)
    # print(DeltaRGB(0,0,0,ord2rgb).shape)
    DRGB = DRGB*DRGB
    DRGB = DRGB.sum(axis = 1)
    min_idx = DRGB.argmin(axis=0)
    # print(ord2rgb[min_idx])
    # print(min_idx)
    return min_idx

def RGB2Item(RGB):
    assert(RGB.shape == (32,32,32,3))
    # return np.ones((32,32,32))
    ord2rgb = np.load("/home/ps/Desktop/DDPM/MC_image/Pix2Vox_Dataset/icons/ord2rgb.npy")
    temp_id = np.arange(253).reshape(253,1)
    ord2rgb = ord2rgb[0:253]
    ord2rgb = np.concatenate((temp_id,ord2rgb),axis = 1)
    # for i in forbidden_items:
    rm_items = set(forbidden_items + non_place_items)
    rm_items = list(rm_items)
    ord2rgb = np.delete(ord2rgb,rm_items,axis=0)
    # print(ord2rgb)

    # print(ord2rgb.shape)
    RGB = RGB.reshape(32,32,32,1,3)
    RGB = np.tile(RGB,(1,1,1,ord2rgb.shape[0],1))
    # print(RGB.shape)
    DRGB = RGB - ord2rgb[:,1:]
    # print(RGB.shape)
    DRGB = DRGB*DRGB
    DRGB = DRGB.sum(axis = -1)
    min_idx = DRGB.argmin(axis = -1).astype(int)
    # print(min_idx.shape)
    return ord2rgb[min_idx][:,:,:,0].astype(int)
    


if __name__ =="__main__":
    temp = np.zeros((32,32,32,3))
    print(RGB2Item(temp).shape)