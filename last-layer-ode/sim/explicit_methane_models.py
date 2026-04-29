import numpy as np

"""
Explicit stoichiometric methane benchmark models.

These functions preserve the published species lists and reaction stoichiometries,
but they intentionally collapse each published reaction into one effective coefficient
k[i] so the API matches your requested format:

    def Model(t, y, k, dim=False)

For reversible published reactions, the code uses one net mass-action term
k[i] * (forward_monomial - reverse_monomial).

So this file is best understood as a stoichiometric, explicit-equation surrogate of the
published mechanisms, not as a thermo-consistent Cantera reimplementation.
"""

def GRI30_FullModel(t: float, y: np.ndarray, k: np.ndarray, dim=False) -> np.ndarray:
    """
    Full 53-species GRI-Mech 3.0 methane / air network written as an explicit stoichiometric ODE system.
    Each published reaction is mapped to one effective mass-action coefficient k[i].
    For reversible reactions we use a net rate k[i]*(forward_monomial - reverse_monomial),
    preserving the published stoichiometry while fitting the compact API you asked for.

    Source: https://combustion.berkeley.edu/gri-mech/version30/text30.html
    Suggested observed species and external controls are returned in dim=True mode.
    """
    if dim == True:
        states = 53
        parameters = 325
        names = ['H2', 'H', 'O', 'O2', 'OH', 'H2O', 'HO2', 'H2O2', 'C', 'CH', 'CH2', 'CH2(S)', 'CH3', 'CH4', 'CO', 'CO2', 'HCO', 'CH2O', 'CH2OH', 'CH3O', 'CH3OH', 'C2H', 'C2H2', 'C2H3', 'C2H4', 'C2H5', 'C2H6', 'HCCO', 'CH2CO', 'HCCOH', 'N', 'NH', 'NH2', 'NH3', 'NNH', 'NO', 'NO2', 'N2O', 'HNO', 'CN', 'HCN', 'H2CN', 'HCNN', 'HCNO', 'HOCN', 'HNCO', 'NCO', 'N2', 'AR', 'C3H7', 'C3H8', 'CH2CHO', 'CH3CHO']
        observed = ['CH4', 'O2', 'CO', 'CO2', 'H2O', 'OH', 'NO']
        inputs = ['feed_CH4', 'feed_O2', 'feed_N2', 'feed_H2O', 'Tin', 'pressure', 'residence_time', 'dilution']
        source = 'https://combustion.berkeley.edu/gri-mech/version30/text30.html'
        return states, parameters, names, observed, inputs, source

    # Unpack species
    (
        H2,
        H,
        O,
        O2,
        OH,
        H2O,
        HO2,
        H2O2,
        C,
        CH,
        CH2,
        CH2_S,
        CH3,
        CH4,
        CO,
        CO2,
        HCO,
        CH2O,
        CH2OH,
        CH3O,
        CH3OH,
        C2H,
        C2H2,
        C2H3,
        C2H4,
        C2H5,
        C2H6,
        HCCO,
        CH2CO,
        HCCOH,
        N,
        NH,
        NH2,
        NH3,
        NNH,
        NO,
        NO2,
        N2O,
        HNO,
        CN,
        HCN,
        H2CN,
        HCNN,
        HCNO,
        HOCN,
        HNCO,
        NCO,
        N2,
        AR,
        C3H7,
        C3H8,
        CH2CHO,
        CH3CHO
    ) = y

    # Unpack effective reaction coefficients
    (
        k1,
        k2,
        k3,
        k4,
        k5,
        k6,
        k7,
        k8,
        k9,
        k10,
        k11,
        k12,
        k13,
        k14,
        k15,
        k16,
        k17,
        k18,
        k19,
        k20,
        k21,
        k22,
        k23,
        k24,
        k25,
        k26,
        k27,
        k28,
        k29,
        k30,
        k31,
        k32,
        k33,
        k34,
        k35,
        k36,
        k37,
        k38,
        k39,
        k40,
        k41,
        k42,
        k43,
        k44,
        k45,
        k46,
        k47,
        k48,
        k49,
        k50,
        k51,
        k52,
        k53,
        k54,
        k55,
        k56,
        k57,
        k58,
        k59,
        k60,
        k61,
        k62,
        k63,
        k64,
        k65,
        k66,
        k67,
        k68,
        k69,
        k70,
        k71,
        k72,
        k73,
        k74,
        k75,
        k76,
        k77,
        k78,
        k79,
        k80,
        k81,
        k82,
        k83,
        k84,
        k85,
        k86,
        k87,
        k88,
        k89,
        k90,
        k91,
        k92,
        k93,
        k94,
        k95,
        k96,
        k97,
        k98,
        k99,
        k100,
        k101,
        k102,
        k103,
        k104,
        k105,
        k106,
        k107,
        k108,
        k109,
        k110,
        k111,
        k112,
        k113,
        k114,
        k115,
        k116,
        k117,
        k118,
        k119,
        k120,
        k121,
        k122,
        k123,
        k124,
        k125,
        k126,
        k127,
        k128,
        k129,
        k130,
        k131,
        k132,
        k133,
        k134,
        k135,
        k136,
        k137,
        k138,
        k139,
        k140,
        k141,
        k142,
        k143,
        k144,
        k145,
        k146,
        k147,
        k148,
        k149,
        k150,
        k151,
        k152,
        k153,
        k154,
        k155,
        k156,
        k157,
        k158,
        k159,
        k160,
        k161,
        k162,
        k163,
        k164,
        k165,
        k166,
        k167,
        k168,
        k169,
        k170,
        k171,
        k172,
        k173,
        k174,
        k175,
        k176,
        k177,
        k178,
        k179,
        k180,
        k181,
        k182,
        k183,
        k184,
        k185,
        k186,
        k187,
        k188,
        k189,
        k190,
        k191,
        k192,
        k193,
        k194,
        k195,
        k196,
        k197,
        k198,
        k199,
        k200,
        k201,
        k202,
        k203,
        k204,
        k205,
        k206,
        k207,
        k208,
        k209,
        k210,
        k211,
        k212,
        k213,
        k214,
        k215,
        k216,
        k217,
        k218,
        k219,
        k220,
        k221,
        k222,
        k223,
        k224,
        k225,
        k226,
        k227,
        k228,
        k229,
        k230,
        k231,
        k232,
        k233,
        k234,
        k235,
        k236,
        k237,
        k238,
        k239,
        k240,
        k241,
        k242,
        k243,
        k244,
        k245,
        k246,
        k247,
        k248,
        k249,
        k250,
        k251,
        k252,
        k253,
        k254,
        k255,
        k256,
        k257,
        k258,
        k259,
        k260,
        k261,
        k262,
        k263,
        k264,
        k265,
        k266,
        k267,
        k268,
        k269,
        k270,
        k271,
        k272,
        k273,
        k274,
        k275,
        k276,
        k277,
        k278,
        k279,
        k280,
        k281,
        k282,
        k283,
        k284,
        k285,
        k286,
        k287,
        k288,
        k289,
        k290,
        k291,
        k292,
        k293,
        k294,
        k295,
        k296,
        k297,
        k298,
        k299,
        k300,
        k301,
        k302,
        k303,
        k304,
        k305,
        k306,
        k307,
        k308,
        k309,
        k310,
        k311,
        k312,
        k313,
        k314,
        k315,
        k316,
        k317,
        k318,
        k319,
        k320,
        k321,
        k322,
        k323,
        k324,
        k325
    ) = k

    # Third-body / falloff effective mixture concentrations
    M1 = 2.4*H2 + H + O + O2 + OH + 15.4*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.75*CO + 3.6*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.83*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M2 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.7*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M12 = 2*H2 + H + O + 6*O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 3.5*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.5*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M33 = H2 + H + O + 0*O2 + OH + 0*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + CH4 + 0.75*CO + 1.5*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 1.5*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + 0*N2 + 0*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M39 = 0*H2 + H + O + O2 + OH + 0*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + CO + 0*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.63*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M43 = 0.73*H2 + H + O + O2 + OH + 3.65*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + CO + CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.38*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M50 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.7*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M52 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 3*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.7*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M54 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.7*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M56 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M57 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M59 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M63 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M70 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.7*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M71 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.7*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M72 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.7*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M74 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.7*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M76 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.7*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M83 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.7*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M85 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.7*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M95 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M131 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.7*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M140 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.7*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M147 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M158 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.7*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M167 = 2*H2 + H + O + O2 + OH + 0*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M174 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.7*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M185 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.625*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M187 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.7*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M205 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.7*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M212 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.7*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M227 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.7*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M230 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.7*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M237 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.7*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M241 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M269 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.7*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M289 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.7*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M304 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.7*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M312 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.7*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M318 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.7*AR + C3H7 + C3H8 + CH2CHO + CH3CHO
    M320 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + C + CH + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH2OH + CH3O + CH3OH + C2H + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + HCCO + CH2CO + HCCOH + N + NH + NH2 + NH3 + NNH + NO + NO2 + N2O + HNO + CN + HCN + H2CN + HCNN + HCNO + HOCN + HNCO + NCO + N2 + 0.7*AR + C3H7 + C3H8 + CH2CHO + CH3CHO

    # Reaction rates
    r1 = k1 * (O**2 - O2) * M1
    r2 = k2 * (O * H - OH) * M2
    r3 = k3 * (O * H2 - H * OH)
    r4 = k4 * (O * HO2 - OH * O2)
    r5 = k5 * (O * H2O2 - OH * HO2)
    r6 = k6 * (O * CH - H * CO)
    r7 = k7 * (O * CH2 - H * HCO)
    r8 = k8 * (O * CH2_S - H2 * CO)
    r9 = k9 * (O * CH2_S - H * HCO)
    r10 = k10 * (O * CH3 - H * CH2O)
    r11 = k11 * (O * CH4 - OH * CH3)
    r12 = k12 * (O * CO - CO2) * M12
    r13 = k13 * (O * HCO - OH * CO)
    r14 = k14 * (O * HCO - H * CO2)
    r15 = k15 * (O * CH2O - OH * HCO)
    r16 = k16 * (O * CH2OH - OH * CH2O)
    r17 = k17 * (O * CH3O - OH * CH2O)
    r18 = k18 * (O * CH3OH - OH * CH2OH)
    r19 = k19 * (O * CH3OH - OH * CH3O)
    r20 = k20 * (O * C2H - CH * CO)
    r21 = k21 * (O * C2H2 - H * HCCO)
    r22 = k22 * (O * C2H2 - OH * C2H)
    r23 = k23 * (O * C2H2 - CO * CH2)
    r24 = k24 * (O * C2H3 - H * CH2CO)
    r25 = k25 * (O * C2H4 - CH3 * HCO)
    r26 = k26 * (O * C2H5 - CH3 * CH2O)
    r27 = k27 * (O * C2H6 - OH * C2H5)
    r28 = k28 * (O * HCCO - H * CO**2)
    r29 = k29 * (O * CH2CO - OH * HCCO)
    r30 = k30 * (O * CH2CO - CH2 * CO2)
    r31 = k31 * (O2 * CO - O * CO2)
    r32 = k32 * (O2 * CH2O - HO2 * HCO)
    r33 = k33 * (H * O2 - HO2) * M33
    r34 = k34 * (H * O2**2 - HO2 * O2)
    r35 = k35 * (H * O2 * H2O - HO2 * H2O)
    r36 = k36 * (H * O2 * N2 - HO2 * N2)
    r37 = k37 * (H * O2 * AR - HO2 * AR)
    r38 = k38 * (H * O2 - O * OH)
    r39 = k39 * (H**2 - H2) * M39
    r40 = k40 * (H**2 * H2 - H2**2)
    r41 = k41 * (H**2 * H2O - H2 * H2O)
    r42 = k42 * (H**2 * CO2 - H2 * CO2)
    r43 = k43 * (H * OH - H2O) * M43
    r44 = k44 * (H * HO2 - O * H2O)
    r45 = k45 * (H * HO2 - O2 * H2)
    r46 = k46 * (H * HO2 - OH**2)
    r47 = k47 * (H * H2O2 - HO2 * H2)
    r48 = k48 * (H * H2O2 - OH * H2O)
    r49 = k49 * (H * CH - C * H2)
    r50 = k50 * (H * CH2 - CH3) * M50
    r51 = k51 * (H * CH2_S - CH * H2)
    r52 = k52 * (H * CH3 - CH4) * M52
    r53 = k53 * (H * CH4 - CH3 * H2)
    r54 = k54 * (H * HCO - CH2O) * M54
    r55 = k55 * (H * HCO - H2 * CO)
    r56 = k56 * (H * CH2O - CH2OH) * M56
    r57 = k57 * (H * CH2O - CH3O) * M57
    r58 = k58 * (H * CH2O - HCO * H2)
    r59 = k59 * (H * CH2OH - CH3OH) * M59
    r60 = k60 * (H * CH2OH - H2 * CH2O)
    r61 = k61 * (H * CH2OH - OH * CH3)
    r62 = k62 * (H * CH2OH - CH2_S * H2O)
    r63 = k63 * (H * CH3O - CH3OH) * M63
    r64 = k64 * (H * CH3O - H * CH2OH)
    r65 = k65 * (H * CH3O - H2 * CH2O)
    r66 = k66 * (H * CH3O - OH * CH3)
    r67 = k67 * (H * CH3O - CH2_S * H2O)
    r68 = k68 * (H * CH3OH - CH2OH * H2)
    r69 = k69 * (H * CH3OH - CH3O * H2)
    r70 = k70 * (H * C2H - C2H2) * M70
    r71 = k71 * (H * C2H2 - C2H3) * M71
    r72 = k72 * (H * C2H3 - C2H4) * M72
    r73 = k73 * (H * C2H3 - H2 * C2H2)
    r74 = k74 * (H * C2H4 - C2H5) * M74
    r75 = k75 * (H * C2H4 - C2H3 * H2)
    r76 = k76 * (H * C2H5 - C2H6) * M76
    r77 = k77 * (H * C2H5 - H2 * C2H4)
    r78 = k78 * (H * C2H6 - C2H5 * H2)
    r79 = k79 * (H * HCCO - CH2_S * CO)
    r80 = k80 * (H * CH2CO - HCCO * H2)
    r81 = k81 * (H * CH2CO - CH3 * CO)
    r82 = k82 * (H * HCCOH - H * CH2CO)
    r83 = k83 * (H2 * CO - CH2O) * M83
    r84 = k84 * (OH * H2 - H * H2O)
    r85 = k85 * (OH**2 - H2O2) * M85
    r86 = k86 * (OH**2 - O * H2O)
    r87 = k87 * (OH * HO2 - O2 * H2O)
    r88 = k88 * (OH * H2O2 - HO2 * H2O)
    r89 = k89 * (OH * H2O2 - HO2 * H2O)
    r90 = k90 * (OH * C - H * CO)
    r91 = k91 * (OH * CH - H * HCO)
    r92 = k92 * (OH * CH2 - H * CH2O)
    r93 = k93 * (OH * CH2 - CH * H2O)
    r94 = k94 * (OH * CH2_S - H * CH2O)
    r95 = k95 * (OH * CH3 - CH3OH) * M95
    r96 = k96 * (OH * CH3 - CH2 * H2O)
    r97 = k97 * (OH * CH3 - CH2_S * H2O)
    r98 = k98 * (OH * CH4 - CH3 * H2O)
    r99 = k99 * (OH * CO - H * CO2)
    r100 = k100 * (OH * HCO - H2O * CO)
    r101 = k101 * (OH * CH2O - HCO * H2O)
    r102 = k102 * (OH * CH2OH - H2O * CH2O)
    r103 = k103 * (OH * CH3O - H2O * CH2O)
    r104 = k104 * (OH * CH3OH - CH2OH * H2O)
    r105 = k105 * (OH * CH3OH - CH3O * H2O)
    r106 = k106 * (OH * C2H - H * HCCO)
    r107 = k107 * (OH * C2H2 - H * CH2CO)
    r108 = k108 * (OH * C2H2 - H * HCCOH)
    r109 = k109 * (OH * C2H2 - C2H * H2O)
    r110 = k110 * (OH * C2H2 - CH3 * CO)
    r111 = k111 * (OH * C2H3 - H2O * C2H2)
    r112 = k112 * (OH * C2H4 - C2H3 * H2O)
    r113 = k113 * (OH * C2H6 - C2H5 * H2O)
    r114 = k114 * (OH * CH2CO - HCCO * H2O)
    r115 = k115 * (HO2**2 - O2 * H2O2)
    r116 = k116 * (HO2**2 - O2 * H2O2)
    r117 = k117 * (HO2 * CH2 - OH * CH2O)
    r118 = k118 * (HO2 * CH3 - O2 * CH4)
    r119 = k119 * (HO2 * CH3 - OH * CH3O)
    r120 = k120 * (HO2 * CO - OH * CO2)
    r121 = k121 * (HO2 * CH2O - HCO * H2O2)
    r122 = k122 * (C * O2 - O * CO)
    r123 = k123 * (C * CH2 - H * C2H)
    r124 = k124 * (C * CH3 - H * C2H2)
    r125 = k125 * (CH * O2 - O * HCO)
    r126 = k126 * (CH * H2 - H * CH2)
    r127 = k127 * (CH * H2O - H * CH2O)
    r128 = k128 * (CH * CH2 - H * C2H2)
    r129 = k129 * (CH * CH3 - H * C2H3)
    r130 = k130 * (CH * CH4 - H * C2H4)
    r131 = k131 * (CH * CO - HCCO) * M131
    r132 = k132 * (CH * CO2 - HCO * CO)
    r133 = k133 * (CH * CH2O - H * CH2CO)
    r134 = k134 * (CH * HCCO - CO * C2H2)
    r135 = k135 * (CH2 * O2)
    r136 = k136 * (CH2 * H2 - H * CH3)
    r137 = k137 * (CH2**2 - H2 * C2H2)
    r138 = k138 * (CH2 * CH3 - H * C2H4)
    r139 = k139 * (CH2 * CH4 - CH3**2)
    r140 = k140 * (CH2 * CO - CH2CO) * M140
    r141 = k141 * (CH2 * HCCO - C2H3 * CO)
    r142 = k142 * (CH2_S * N2 - CH2 * N2)
    r143 = k143 * (CH2_S * AR - CH2 * AR)
    r144 = k144 * (CH2_S * O2 - H * OH * CO)
    r145 = k145 * (CH2_S * O2 - CO * H2O)
    r146 = k146 * (CH2_S * H2 - CH3 * H)
    r147 = k147 * (CH2_S * H2O - CH3OH) * M147
    r148 = k148 * (CH2_S * H2O - CH2 * H2O)
    r149 = k149 * (CH2_S * CH3 - H * C2H4)
    r150 = k150 * (CH2_S * CH4 - CH3**2)
    r151 = k151 * (CH2_S * CO - CH2 * CO)
    r152 = k152 * (CH2_S * CO2 - CH2 * CO2)
    r153 = k153 * (CH2_S * CO2 - CO * CH2O)
    r154 = k154 * (CH2_S * C2H6 - CH3 * C2H5)
    r155 = k155 * (CH3 * O2 - O * CH3O)
    r156 = k156 * (CH3 * O2 - OH * CH2O)
    r157 = k157 * (CH3 * H2O2 - HO2 * CH4)
    r158 = k158 * (CH3**2 - C2H6) * M158
    r159 = k159 * (CH3**2 - H * C2H5)
    r160 = k160 * (CH3 * HCO - CH4 * CO)
    r161 = k161 * (CH3 * CH2O - HCO * CH4)
    r162 = k162 * (CH3 * CH3OH - CH2OH * CH4)
    r163 = k163 * (CH3 * CH3OH - CH3O * CH4)
    r164 = k164 * (CH3 * C2H4 - C2H3 * CH4)
    r165 = k165 * (CH3 * C2H6 - C2H5 * CH4)
    r166 = k166 * (HCO * H2O - H * CO * H2O)
    r167 = k167 * (HCO - H * CO) * M167
    r168 = k168 * (HCO * O2 - HO2 * CO)
    r169 = k169 * (CH2OH * O2 - HO2 * CH2O)
    r170 = k170 * (CH3O * O2 - HO2 * CH2O)
    r171 = k171 * (C2H * O2 - HCO * CO)
    r172 = k172 * (C2H * H2 - H * C2H2)
    r173 = k173 * (C2H3 * O2 - HCO * CH2O)
    r174 = k174 * (C2H4 - H2 * C2H2) * M174
    r175 = k175 * (C2H5 * O2 - HO2 * C2H4)
    r176 = k176 * (HCCO * O2 - OH * CO**2)
    r177 = k177 * (HCCO**2 - CO**2 * C2H2)
    r178 = k178 * (N * NO - N2 * O)
    r179 = k179 * (N * O2 - NO * O)
    r180 = k180 * (N * OH - NO * H)
    r181 = k181 * (N2O * O - N2 * O2)
    r182 = k182 * (N2O * O - NO**2)
    r183 = k183 * (N2O * H - N2 * OH)
    r184 = k184 * (N2O * OH - N2 * HO2)
    r185 = k185 * (N2O - N2 * O) * M185
    r186 = k186 * (HO2 * NO - NO2 * OH)
    r187 = k187 * (NO * O - NO2) * M187
    r188 = k188 * (NO2 * O - NO * O2)
    r189 = k189 * (NO2 * H - NO * OH)
    r190 = k190 * (NH * O - NO * H)
    r191 = k191 * (NH * H - N * H2)
    r192 = k192 * (NH * OH - HNO * H)
    r193 = k193 * (NH * OH - N * H2O)
    r194 = k194 * (NH * O2 - HNO * O)
    r195 = k195 * (NH * O2 - NO * OH)
    r196 = k196 * (NH * N - N2 * H)
    r197 = k197 * (NH * H2O - HNO * H2)
    r198 = k198 * (NH * NO - N2 * OH)
    r199 = k199 * (NH * NO - N2O * H)
    r200 = k200 * (NH2 * O - OH * NH)
    r201 = k201 * (NH2 * O - H * HNO)
    r202 = k202 * (NH2 * H - NH * H2)
    r203 = k203 * (NH2 * OH - NH * H2O)
    r204 = k204 * (NNH - N2 * H)
    r205 = k205 * (NNH - N2 * H) * M205
    r206 = k206 * (NNH * O2 - HO2 * N2)
    r207 = k207 * (NNH * O - OH * N2)
    r208 = k208 * (NNH * O - NH * NO)
    r209 = k209 * (NNH * H - H2 * N2)
    r210 = k210 * (NNH * OH - H2O * N2)
    r211 = k211 * (NNH * CH3 - CH4 * N2)
    r212 = k212 * (H * NO - HNO) * M212
    r213 = k213 * (HNO * O - NO * OH)
    r214 = k214 * (HNO * H - H2 * NO)
    r215 = k215 * (HNO * OH - NO * H2O)
    r216 = k216 * (HNO * O2 - HO2 * NO)
    r217 = k217 * (CN * O - CO * N)
    r218 = k218 * (CN * OH - NCO * H)
    r219 = k219 * (CN * H2O - HCN * OH)
    r220 = k220 * (CN * O2 - NCO * O)
    r221 = k221 * (CN * H2 - HCN * H)
    r222 = k222 * (NCO * O - NO * CO)
    r223 = k223 * (NCO * H - NH * CO)
    r224 = k224 * (NCO * OH - NO * H * CO)
    r225 = k225 * (NCO * N - N2 * CO)
    r226 = k226 * (NCO * O2 - NO * CO2)
    r227 = k227 * (NCO - N * CO) * M227
    r228 = k228 * (NCO * NO - N2O * CO)
    r229 = k229 * (NCO * NO - N2 * CO2)
    r230 = k230 * (HCN - H * CN) * M230
    r231 = k231 * (HCN * O - NCO * H)
    r232 = k232 * (HCN * O - NH * CO)
    r233 = k233 * (HCN * O - CN * OH)
    r234 = k234 * (HCN * OH - HOCN * H)
    r235 = k235 * (HCN * OH - HNCO * H)
    r236 = k236 * (HCN * OH - NH2 * CO)
    r237 = k237 * (H * HCN - H2CN) * M237
    r238 = k238 * (H2CN * N - N2 * CH2)
    r239 = k239 * (C * N2 - CN * N)
    r240 = k240 * (CH * N2 - HCN * N)
    r241 = k241 * (CH * N2 - HCNN) * M241
    r242 = k242 * (CH2 * N2 - HCN * NH)
    r243 = k243 * (CH2_S * N2 - NH * HCN)
    r244 = k244 * (C * NO - CN * O)
    r245 = k245 * (C * NO - CO * N)
    r246 = k246 * (CH * NO - HCN * O)
    r247 = k247 * (CH * NO - H * NCO)
    r248 = k248 * (CH * NO - N * HCO)
    r249 = k249 * (CH2 * NO - H * HNCO)
    r250 = k250 * (CH2 * NO - OH * HCN)
    r251 = k251 * (CH2 * NO - H * HCNO)
    r252 = k252 * (CH2_S * NO - H * HNCO)
    r253 = k253 * (CH2_S * NO - OH * HCN)
    r254 = k254 * (CH2_S * NO - H * HCNO)
    r255 = k255 * (CH3 * NO - HCN * H2O)
    r256 = k256 * (CH3 * NO - H2CN * OH)
    r257 = k257 * (HCNN * O - CO * H * N2)
    r258 = k258 * (HCNN * O - HCN * NO)
    r259 = k259 * (HCNN * O2 - O * HCO * N2)
    r260 = k260 * (HCNN * OH - H * HCO * N2)
    r261 = k261 * (HCNN * H - CH2 * N2)
    r262 = k262 * (HNCO * O - NH * CO2)
    r263 = k263 * (HNCO * O - HNO * CO)
    r264 = k264 * (HNCO * O - NCO * OH)
    r265 = k265 * (HNCO * H - NH2 * CO)
    r266 = k266 * (HNCO * H - H2 * NCO)
    r267 = k267 * (HNCO * OH - NCO * H2O)
    r268 = k268 * (HNCO * OH - NH2 * CO2)
    r269 = k269 * (HNCO - NH * CO) * M269
    r270 = k270 * (HCNO * H - H * HNCO)
    r271 = k271 * (HCNO * H - OH * HCN)
    r272 = k272 * (HCNO * H - NH2 * CO)
    r273 = k273 * (HOCN * H - H * HNCO)
    r274 = k274 * (HCCO * NO - HCNO * CO)
    r275 = k275 * (CH3 * N - H2CN * H)
    r276 = k276 * (CH3 * N - HCN * H2)
    r277 = k277 * (NH3 * H - NH2 * H2)
    r278 = k278 * (NH3 * OH - NH2 * H2O)
    r279 = k279 * (NH3 * O - NH2 * OH)
    r280 = k280 * (NH * CO2 - HNO * CO)
    r281 = k281 * (CN * NO2 - NCO * NO)
    r282 = k282 * (NCO * NO2 - N2O * CO2)
    r283 = k283 * (N * CO2 - NO * CO)
    r284 = k284 * (O * CH3)
    r285 = k285 * (O * C2H4 - H * CH2CHO)
    r286 = k286 * (O * C2H5 - H * CH3CHO)
    r287 = k287 * (OH * HO2 - O2 * H2O)
    r288 = k288 * (OH * CH3)
    r289 = k289 * (CH * H2 - CH3) * M289
    r290 = k290 * (CH2 * O2)
    r291 = k291 * (CH2 * O2 - O * CH2O)
    r292 = k292 * (CH2**2)
    r293 = k293 * (CH2_S * H2O)
    r294 = k294 * (C2H3 * O2 - O * CH2CHO)
    r295 = k295 * (C2H3 * O2 - HO2 * C2H2)
    r296 = k296 * (O * CH3CHO - OH * CH2CHO)
    r297 = k297 * (O * CH3CHO)
    r298 = k298 * (O2 * CH3CHO)
    r299 = k299 * (H * CH3CHO - CH2CHO * H2)
    r300 = k300 * (H * CH3CHO)
    r301 = k301 * (OH * CH3CHO)
    r302 = k302 * (HO2 * CH3CHO)
    r303 = k303 * (CH3 * CH3CHO)
    r304 = k304 * (H * CH2CO - CH2CHO) * M304
    r305 = k305 * (O * CH2CHO)
    r306 = k306 * (O2 * CH2CHO)
    r307 = k307 * (O2 * CH2CHO)
    r308 = k308 * (H * CH2CHO - CH3 * HCO)
    r309 = k309 * (H * CH2CHO - CH2CO * H2)
    r310 = k310 * (OH * CH2CHO - H2O * CH2CO)
    r311 = k311 * (OH * CH2CHO - HCO * CH2OH)
    r312 = k312 * (CH3 * C2H5 - C3H8) * M312
    r313 = k313 * (O * C3H8 - OH * C3H7)
    r314 = k314 * (H * C3H8 - C3H7 * H2)
    r315 = k315 * (OH * C3H8 - C3H7 * H2O)
    r316 = k316 * (C3H7 * H2O2 - HO2 * C3H8)
    r317 = k317 * (CH3 * C3H8 - C3H7 * CH4)
    r318 = k318 * (CH3 * C2H4 - C3H7) * M318
    r319 = k319 * (O * C3H7 - C2H5 * CH2O)
    r320 = k320 * (H * C3H7 - C3H8) * M320
    r321 = k321 * (H * C3H7 - CH3 * C2H5)
    r322 = k322 * (OH * C3H7 - C2H5 * CH2OH)
    r323 = k323 * (HO2 * C3H7 - O2 * C3H8)
    r324 = k324 * (HO2 * C3H7)
    r325 = k325 * (CH3 * C3H7 - C2H5**2)

    # Species balances
    dH2 = - r3 + r8 + r39 + r40 + r41 + r42 + r45 + r47 + r49 + r51 + r53 + r55 + r58 + r60 + r65 + r68 + r69 \
        + r73 + r75 + r77 + r78 + r80 - r83 - r84 - r126 - r136 + r137 - r146 - r172 + r174 + r191 + r197 + r202 \
        + r209 + r214 - r221 + r266 + r276 + r277 + r284 + r288 - r289 + r293 + r299 + r300 + r309 + r314
    dH = - r2 + r3 + r6 + r7 + r9 + r10 + r14 + r21 + r24 + r28 - r33 - r34 - r35 - r36 - r37 - r38 - 2*r39 \
        - 2*r40 - 2*r41 - 2*r42 - r43 - r44 - r45 - r46 - r47 - r48 - r49 - r50 - r51 - r52 - r53 - r54 - r55 \
        - r56 - r57 - r58 - r59 - r60 - r61 - r62 - r63 - r65 - r66 - r67 - r68 - r69 - r70 - r71 - r72 - r73 \
        - r74 - r75 - r76 - r77 - r78 - r79 - r80 - r81 + r84 + r90 + r91 + r92 + r94 + r99 + r106 + r107 + r108 \
        + r123 + r124 + r126 + r127 + r128 + r129 + r130 + r133 + r135 + r136 + r138 + r144 + r146 + r149 + r159 \
        + r166 + r167 + r172 + r180 - r183 - r189 + r190 - r191 + r192 + r196 + r199 + r201 - r202 + r204 + r205 \
        - r209 - r212 - r214 + r218 + r221 - r223 + r224 + r230 + r231 + r234 + r235 - r237 + r247 + r249 + r251 \
        + r252 + r254 + r257 + r260 - r261 - r265 - r266 - r271 - r272 + r275 - r277 + r284 + r285 + r286 \
        + 2*r290 + 2*r292 - r299 - r300 - r304 + r305 - r308 - r309 - r314 - r320 - r321
    dO = - 2*r1 - r2 - r3 - r4 - r5 - r6 - r7 - r8 - r9 - r10 - r11 - r12 - r13 - r14 - r15 - r16 - r17 - r18 \
        - r19 - r20 - r21 - r22 - r23 - r24 - r25 - r26 - r27 - r28 - r29 - r30 + r31 + r38 + r44 + r86 + r122 \
        + r125 + r155 + r178 + r179 - r181 - r182 + r185 - r187 - r188 - r190 + r194 - r200 - r201 - r207 - r208 \
        - r213 - r217 + r220 - r222 - r231 - r232 - r233 + r244 + r246 - r257 - r258 + r259 - r262 - r263 - r264 \
        - r279 - r284 - r285 - r286 + r291 + r294 - r296 - r297 - r305 - r313 - r319
    dO2 = r1 + r4 - r31 - r32 - r33 - r34 - r35 - r36 - r37 - r38 + r45 + r87 + r115 + r116 + r118 - r122 - r125 \
        - r135 - r144 - r145 - r155 - r156 - r168 - r169 - r170 - r171 - r173 - r175 - r176 - r179 + r181 + r188 \
        - r194 - r195 - r206 - r216 - r220 - r226 - r259 + r287 - r290 - r291 - r294 - r295 - r298 - r306 - r307 \
        + r323
    dOH = r2 + r3 + r4 + r5 + r11 + r13 + r15 + r16 + r17 + r18 + r19 + r22 + r27 + r29 + r38 - r43 + 2*r46 + r48 \
        + r61 + r66 - r84 - 2*r85 - 2*r86 - r87 - r88 - r89 - r90 - r91 - r92 - r93 - r94 - r95 - r96 - r97 \
        - r98 - r99 - r100 - r101 - r102 - r103 - r104 - r105 - r106 - r107 - r108 - r109 - r110 - r111 - r112 \
        - r113 - r114 + r117 + r119 + r120 + r135 + r144 + r156 + r176 - r180 + r183 - r184 + r186 + r189 - r192 \
        - r193 + r195 + r198 + r200 - r203 + r207 - r210 + r213 - r215 - r218 + r219 - r224 + r233 - r234 - r235 \
        - r236 + r250 + r253 + r256 - r260 + r264 - r267 - r268 + r271 - r278 + r279 - r287 - r288 + r296 + r297 \
        - r301 + r306 + r307 - r310 - r311 + r313 - r315 - r322 + r324
    dH2O = r43 + r44 + r48 + r62 + r67 + r84 + r86 + r87 + r88 + r89 + r93 + r96 + r97 + r98 + r100 + r101 + r102 \
        + r103 + r104 + r105 + r109 + r111 + r112 + r113 + r114 - r127 + r145 - r147 + r193 - r197 + r203 + r210 \
        + r215 - r219 + r255 + r267 + r278 + r287 - r293 + r301 + r310 + r315
    dHO2 = - r4 + r5 + r32 + r33 + r34 + r35 + r36 + r37 - r44 - r45 - r46 + r47 - r87 + r88 + r89 - 2*r115 \
        - 2*r116 - r117 - r118 - r119 - r120 - r121 + r157 + r168 + r169 + r170 + r175 + r184 - r186 + r206 \
        + r216 - r287 + r295 + r298 - r302 + r316 - r323 - r324
    dH2O2 = - r5 - r47 - r48 + r85 - r88 - r89 + r115 + r116 + r121 - r157 + r302 - r316
    dC = r49 - r90 - r122 - r123 - r124 - r239 - r244 - r245
    dCH = - r6 + r20 - r49 + r51 - r91 + r93 - r125 - r126 - r127 - r128 - r129 - r130 - r131 - r132 - r133 - r134 \
        - r240 - r241 - r246 - r247 - r248 - r289
    dCH2 = - r7 + r23 + r30 - r50 - r92 - r93 + r96 - r117 - r123 + r126 - r128 - r135 - r136 - 2*r137 - r138 \
        - r139 - r140 - r141 + r142 + r143 + r148 + r151 + r152 + r238 - r242 - r249 - r250 - r251 + r261 - r290 \
        - r291 - 2*r292 + r305
    dCH2_S = - r8 - r9 - r51 + r62 + r67 + r79 - r94 + r97 - r142 - r143 - r144 - r145 - r146 - r147 - r148 - r149 \
        - r150 - r151 - r152 - r153 - r154 - r243 - r252 - r253 - r254 - r293
    dCH3 = - r10 + r11 + r25 + r26 + r50 - r52 + r53 + r61 + r66 + r81 - r95 - r96 - r97 + r98 + r110 - r118 - r119 \
        - r124 - r129 + r136 - r138 + 2*r139 + r146 - r149 + 2*r150 + r154 - r155 - r156 - r157 - 2*r158 \
        - 2*r159 - r160 - r161 - r162 - r163 - r164 - r165 - r211 - r255 - r256 - r275 - r276 - r284 - r288 \
        + r289 + r297 + r298 + r300 + r301 + r302 + r308 - r312 - r317 - r318 + r321 - r325
    dCH4 = - r11 + r52 - r53 - r98 + r118 - r130 - r139 - r150 + r157 + r160 + r161 + r162 + r163 + r164 + r165 \
        + r211 + r303 + r317
    dCO = r6 + r8 - r12 + r13 + r20 + r23 + 2*r28 - r31 + r55 + r79 + r81 - r83 + r90 - r99 + r100 + r110 - r120 \
        + r122 - r131 + r132 + r134 + r135 - r140 + r141 + r144 + r145 + r153 + r160 + r166 + r167 + r168 + r171 \
        + 2*r176 + 2*r177 + r217 + r222 + r223 + r224 + r225 + r227 + r228 + r232 + r236 + r245 + r257 + r263 \
        + r265 + r269 + r272 + r274 + r280 + r283 + r284 + r297 + r298 + r300 + r301 + r302 + r303 + r306
    dCO2 = r12 + r14 + r30 + r31 + r99 + r120 - r132 - r153 + r226 + r229 + r262 + r268 - r280 + r282 - r283 + r290 \
        + r305
    dHCO = r7 + r9 - r13 - r14 + r15 + r25 + r32 - r54 - r55 + r58 + r91 - r100 + r101 + r121 + r125 + r132 - r160 \
        + r161 - r166 - r167 - r168 + r171 + r173 + r248 + r259 + r260 + 2*r307 + r308 + r311
    dCH2O = r10 - r15 + r16 + r17 + r26 - r32 + r54 - r56 - r57 - r58 + r60 + r65 + r83 + r92 + r94 - r101 + r102 \
        + r103 + r117 - r121 + r127 - r133 + r153 + r156 - r161 + r169 + r170 + r173 + r288 + r291 + r293 + r306 \
        + r319 + r324
    dCH2OH = - r16 + r18 + r56 - r59 - r60 - r61 - r62 + r64 + r68 - r102 + r104 + r162 - r169 + r311 + r322
    dCH3O = - r17 + r19 + r57 - r63 - r64 - r65 - r66 - r67 + r69 - r103 + r105 + r119 + r155 + r163 - r170
    dCH3OH = - r18 - r19 + r59 + r63 - r68 - r69 + r95 - r104 - r105 + r147 - r162 - r163
    dC2H = - r20 + r22 - r70 - r106 + r109 + r123 - r171 - r172
    dC2H2 = - r21 - r22 - r23 + r70 - r71 + r73 - r107 - r108 - r109 - r110 + r111 + r124 + r128 + r134 + r137 \
        + r172 + r174 + r177 + r292 + r295
    dC2H3 = - r24 + r71 - r72 - r73 + r75 - r111 + r112 + r129 + r141 + r164 - r173 - r294 - r295
    dC2H4 = - r25 + r72 - r74 - r75 + r77 - r112 + r130 + r138 + r149 - r164 - r174 + r175 - r285 - r318
    dC2H5 = - r26 + r27 + r74 - r76 - r77 + r78 + r113 + r154 + r159 + r165 - r175 - r286 - r312 + r319 + r321 \
        + r322 + r324 + 2*r325
    dC2H6 = - r27 + r76 - r78 - r113 - r154 + r158 - r165
    dHCCO = r21 - r28 + r29 - r79 + r80 + r106 + r114 + r131 - r134 - r141 - r176 - 2*r177 - r274
    dCH2CO = r24 - r29 - r30 - r80 - r81 + r82 + r107 - r114 + r133 + r140 - r304 + r309 + r310
    dHCCOH = - r82 + r108
    dN = - r178 - r179 - r180 + r191 + r193 - r196 + r217 - r225 + r227 - r238 + r239 + r240 + r245 + r248 - r275 \
        - r276 - r283
    dNH = - r190 - r191 - r192 - r193 - r194 - r195 - r196 - r197 - r198 - r199 + r200 + r202 + r203 + r208 + r223 \
        + r232 + r242 + r243 + r262 + r269 - r280
    dNH2 = - r200 - r201 - r202 - r203 + r236 + r265 + r268 + r272 + r277 + r278 + r279
    dNH3 = - r277 - r278 - r279
    dNNH = - r204 - r205 - r206 - r207 - r208 - r209 - r210 - r211
    dNO = - r178 + r179 + r180 + 2*r182 - r186 - r187 + r188 + r189 + r190 + r195 - r198 - r199 + r208 - r212 \
        + r213 + r214 + r215 + r216 + r222 + r224 + r226 - r228 - r229 - r244 - r245 - r246 - r247 - r248 - r249 \
        - r250 - r251 - r252 - r253 - r254 - r255 - r256 + r258 - r274 + r281 + r283
    dNO2 = r186 + r187 - r188 - r189 - r281 - r282
    dN2O = - r181 - r182 - r183 - r184 - r185 + r199 + r228 + r282
    dHNO = r192 + r194 + r197 + r201 + r212 - r213 - r214 - r215 - r216 + r263 + r280
    dCN = - r217 - r218 - r219 - r220 - r221 + r230 + r233 + r239 + r244 - r281
    dHCN = r219 + r221 - r230 - r231 - r232 - r233 - r234 - r235 - r236 - r237 + r240 + r242 + r243 + r246 + r250 \
        + r253 + r255 + r258 + r271 + r276
    dH2CN = r237 - r238 + r256 + r275
    dHCNN = r241 - r257 - r258 - r259 - r260 - r261
    dHCNO = r251 + r254 - r270 - r271 - r272 + r274
    dHOCN = r234 - r273
    dHNCO = r235 + r249 + r252 - r262 - r263 - r264 - r265 - r266 - r267 - r268 - r269 + r270 + r273
    dNCO = r218 + r220 - r222 - r223 - r224 - r225 - r226 - r227 - r228 - r229 + r231 + r247 + r264 + r266 + r267 \
        + r281 - r282
    dN2 = r178 + r181 + r183 + r184 + r185 + r196 + r198 + r204 + r205 + r206 + r207 + r209 + r210 + r211 + r225 \
        + r229 + r238 - r239 - r240 - r241 - r242 - r243 + r257 + r259 + r260 + r261
    dAR = 0.0
    dC3H7 = r313 + r314 + r315 - r316 + r317 + r318 - r319 - r320 - r321 - r322 - r323 - r324 - r325
    dC3H8 = r312 - r313 - r314 - r315 + r316 - r317 + r320 + r323
    dCH2CHO = r285 + r294 + r296 + r299 + r304 - r305 - r306 - r307 - r308 - r309 - r310 - r311
    dCH3CHO = r286 - r296 - r297 - r298 - r299 - r300 - r301 - r302 - r303

    return np.array([dH2, dH, dO, dO2, dOH, dH2O, dHO2, dH2O2, dC, dCH, dCH2, dCH2_S, dCH3, dCH4, dCO, dCO2, dHCO, dCH2O, dCH2OH, dCH3O, dCH3OH, dC2H, dC2H2, dC2H3, dC2H4, dC2H5, dC2H6, dHCCO, dCH2CO, dHCCOH, dN, dNH, dNH2, dNH3, dNNH, dNO, dNO2, dN2O, dHNO, dCN, dHCN, dH2CN, dHCNN, dHCNO, dHOCN, dHNCO, dNCO, dN2, dAR, dC3H7, dC3H8, dCH2CHO, dCH3CHO], dtype=float)


def Kazakov_MiddleModel(t: float, y: np.ndarray, k: np.ndarray, dim=False) -> np.ndarray:
    """
    Middle-size public methane mechanism from the CollectionOfMechanisms repository.
    The public YAML conversion expands to 28 species and 116 reactions, and this function
    follows that exact public YAML rather than the shorter folder label.

    Source: https://raw.githubusercontent.com/jiweiqi/CollectionOfMechanisms/master/CH4_Methane/CH4_Kazakov_s22r104/CH4_Kazakov_s22r104.yaml
    Suggested observed species and external controls are returned in dim=True mode.
    """
    if dim == True:
        states = 28
        parameters = 116
        names = ['H2', 'H', 'O', 'O2', 'OH', 'H2O', 'HO2', 'H2O2', 'CH2', 'CH2-S', 'CH3', 'CH4', 'CO', 'CO2', 'HCO', 'CH2O', 'CH3O', 'C2H2', 'C2H3', 'C2H4', 'C2H5', 'C2H6', 'N2', 'AR', 'NO', 'NO2', 'N2O', 'N']
        observed = ['CH4', 'O2', 'CO', 'CO2', 'H2O', 'OH', 'NO']
        inputs = ['feed_CH4', 'feed_O2', 'feed_N2', 'feed_H2O', 'Tin', 'pressure', 'residence_time', 'dilution']
        source = 'https://raw.githubusercontent.com/jiweiqi/CollectionOfMechanisms/master/CH4_Methane/CH4_Kazakov_s22r104/CH4_Kazakov_s22r104.yaml'
        return states, parameters, names, observed, inputs, source

    # Unpack species
    (
        H2,
        H,
        O,
        O2,
        OH,
        H2O,
        HO2,
        H2O2,
        CH2,
        CH2_S,
        CH3,
        CH4,
        CO,
        CO2,
        HCO,
        CH2O,
        CH3O,
        C2H2,
        C2H3,
        C2H4,
        C2H5,
        C2H6,
        N2,
        AR,
        NO,
        NO2,
        N2O,
        N
    ) = y

    # Unpack effective reaction coefficients
    (
        k1,
        k2,
        k3,
        k4,
        k5,
        k6,
        k7,
        k8,
        k9,
        k10,
        k11,
        k12,
        k13,
        k14,
        k15,
        k16,
        k17,
        k18,
        k19,
        k20,
        k21,
        k22,
        k23,
        k24,
        k25,
        k26,
        k27,
        k28,
        k29,
        k30,
        k31,
        k32,
        k33,
        k34,
        k35,
        k36,
        k37,
        k38,
        k39,
        k40,
        k41,
        k42,
        k43,
        k44,
        k45,
        k46,
        k47,
        k48,
        k49,
        k50,
        k51,
        k52,
        k53,
        k54,
        k55,
        k56,
        k57,
        k58,
        k59,
        k60,
        k61,
        k62,
        k63,
        k64,
        k65,
        k66,
        k67,
        k68,
        k69,
        k70,
        k71,
        k72,
        k73,
        k74,
        k75,
        k76,
        k77,
        k78,
        k79,
        k80,
        k81,
        k82,
        k83,
        k84,
        k85,
        k86,
        k87,
        k88,
        k89,
        k90,
        k91,
        k92,
        k93,
        k94,
        k95,
        k96,
        k97,
        k98,
        k99,
        k100,
        k101,
        k102,
        k103,
        k104,
        k105,
        k106,
        k107,
        k108,
        k109,
        k110,
        k111,
        k112,
        k113,
        k114,
        k115,
        k116
    ) = k

    # Third-body / falloff effective mixture concentrations
    M1 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH3O + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + N2 + 0.7*AR + NO + NO2 + N2O + N
    M8 = 2*H2 + H + O + 6*O2 + OH + 6*H2O + HO2 + H2O2 + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 3.5*CO2 + HCO + CH2O + CH3O + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + N2 + 0.5*AR + NO + NO2 + N2O + N
    M19 = H2 + H + O + 0*O2 + OH + 0*H2O + HO2 + H2O2 + CH2 + CH2_S + CH3 + CH4 + 0.75*CO + 1.5*CO2 + HCO + CH2O + CH3O + C2H2 + C2H3 + C2H4 + C2H5 + 1.5*C2H6 + 0*N2 + 0*AR + NO + NO2 + N2O + N
    M25 = 0*H2 + H + O + O2 + OH + 0*H2O + HO2 + H2O2 + CH2 + CH2_S + CH3 + 2*CH4 + CO + 0*CO2 + HCO + CH2O + CH3O + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + N2 + 0.63*AR + NO + NO2 + N2O + N
    M29 = 0.73*H2 + H + O + O2 + OH + 3.65*H2O + HO2 + H2O2 + CH2 + CH2_S + CH3 + 2*CH4 + CO + CO2 + HCO + CH2O + CH3O + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + N2 + 0.38*AR + NO + NO2 + N2O + N
    M33 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH3O + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + N2 + 0.7*AR + NO + NO2 + N2O + N
    M34 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH3O + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + N2 + 0.7*AR + NO + NO2 + N2O + N
    M36 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH3O + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + N2 + 0.7*AR + NO + NO2 + N2O + N
    M38 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH3O + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + N2 + AR + NO + NO2 + N2O + N
    M41 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH3O + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + N2 + 0.7*AR + NO + NO2 + N2O + N
    M42 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH3O + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + N2 + 0.7*AR + NO + NO2 + N2O + N
    M44 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH3O + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + N2 + 0.7*AR + NO + NO2 + N2O + N
    M46 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH3O + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + N2 + 0.7*AR + NO + NO2 + N2O + N
    M48 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH3O + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + N2 + 0.7*AR + NO + NO2 + N2O + N
    M50 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH3O + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + N2 + 0.7*AR + NO + NO2 + N2O + N
    M92 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH3O + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + N2 + 0.7*AR + NO + NO2 + N2O + N
    M99 = 2*H2 + H + O + O2 + OH + 0*H2O + HO2 + H2O2 + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH3O + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + N2 + AR + NO + NO2 + N2O + N
    M103 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH3O + C2H2 + C2H3 + C2H4 + C2H5 + 3*C2H6 + N2 + 0.7*AR + NO + NO2 + N2O + N
    M112 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH3O + C2H2 + C2H3 + C2H4 + C2H5 + C2H6 + N2 + AR + NO + NO2 + N2O + N
    M114 = 2*H2 + H + O + O2 + OH + 6*H2O + HO2 + H2O2 + CH2 + CH2_S + CH3 + 2*CH4 + 1.5*CO + 2*CO2 + HCO + CH2O + CH3O + C2H2 + C2H3 + C2H4 + C2H5 + C2H6 + N2 + AR + NO + NO2 + N2O + N

    # Reaction rates
    r1 = k1 * (O * H - OH) * M1
    r2 = k2 * (O * H2 - H * OH)
    r3 = k3 * (O * HO2 - OH * O2)
    r4 = k4 * (O * CH2 - H * HCO)
    r5 = k5 * (O * CH2_S - H * HCO)
    r6 = k6 * (O * CH3 - H * CH2O)
    r7 = k7 * (O * CH4 - OH * CH3)
    r8 = k8 * (O * CO - CO2) * M8
    r9 = k9 * (O * HCO - OH * CO)
    r10 = k10 * (O * HCO - H * CO2)
    r11 = k11 * (O * CH2O - OH * HCO)
    r12 = k12 * (O * C2H2 - CH2_S * CO)
    r13 = k13 * (O * C2H2 - CO * CH2)
    r14 = k14 * (O * C2H4 - CH3 * HCO)
    r15 = k15 * (O * C2H5 - CH3 * CH2O)
    r16 = k16 * (O * C2H6 - OH * C2H5)
    r17 = k17 * (O2 * CO - O * CO2)
    r18 = k18 * (O2 * CH2O - HO2 * HCO)
    r19 = k19 * (H * O2 - HO2) * M19
    r20 = k20 * (H * O2**2 - HO2 * O2)
    r21 = k21 * (H * O2 * H2O - HO2 * H2O)
    r22 = k22 * (H * O2 * N2 - HO2 * N2)
    r23 = k23 * (H * O2 * AR - HO2 * AR)
    r24 = k24 * (H * O2 - O * OH)
    r25 = k25 * (H**2 - H2) * M25
    r26 = k26 * (H**2 * H2 - H2**2)
    r27 = k27 * (H**2 * H2O - H2 * H2O)
    r28 = k28 * (H**2 * CO2 - H2 * CO2)
    r29 = k29 * (H * OH - H2O) * M29
    r30 = k30 * (H * HO2 - O2 * H2)
    r31 = k31 * (H * HO2 - OH**2)
    r32 = k32 * (H * H2O2 - HO2 * H2)
    r33 = k33 * (H * CH2 - CH3) * M33
    r34 = k34 * (H * CH3 - CH4) * M34
    r35 = k35 * (H * CH4 - CH3 * H2)
    r36 = k36 * (H * HCO - CH2O) * M36
    r37 = k37 * (H * HCO - H2 * CO)
    r38 = k38 * (H * CH2O - CH3O) * M38
    r39 = k39 * (H * CH2O - HCO * H2)
    r40 = k40 * (H * CH3O - OH * CH3)
    r41 = k41 * (H * C2H2 - C2H3) * M41
    r42 = k42 * (H * C2H3 - C2H4) * M42
    r43 = k43 * (H * C2H3 - H2 * C2H2)
    r44 = k44 * (H * C2H4 - C2H5) * M44
    r45 = k45 * (H * C2H4 - C2H3 * H2)
    r46 = k46 * (H * C2H5 - C2H6) * M46
    r47 = k47 * (H * C2H6 - C2H5 * H2)
    r48 = k48 * (H2 * CO - CH2O) * M48
    r49 = k49 * (OH * H2 - H * H2O)
    r50 = k50 * (OH**2 - H2O2) * M50
    r51 = k51 * (OH**2 - O * H2O)
    r52 = k52 * (OH * HO2 - O2 * H2O)
    r53 = k53 * (OH * H2O2 - HO2 * H2O)
    r54 = k54 * (OH * CH2 - H * CH2O)
    r55 = k55 * (OH * CH2_S - H * CH2O)
    r56 = k56 * (OH * CH3 - CH2 * H2O)
    r57 = k57 * (OH * CH3 - CH2_S * H2O)
    r58 = k58 * (OH * CH4 - CH3 * H2O)
    r59 = k59 * (OH * CO - H * CO2)
    r60 = k60 * (OH * HCO - H2O * CO)
    r61 = k61 * (OH * CH2O - HCO * H2O)
    r62 = k62 * (OH * C2H2 - CH3 * CO)
    r63 = k63 * (OH * C2H3 - H2O * C2H2)
    r64 = k64 * (OH * C2H4 - C2H3 * H2O)
    r65 = k65 * (OH * C2H6 - C2H5 * H2O)
    r66 = k66 * (HO2**2 - O2 * H2O2)
    r67 = k67 * (HO2**2 - O2 * H2O2)
    r68 = k68 * (HO2 * CH2 - OH * CH2O)
    r69 = k69 * (HO2 * CH3 - O2 * CH4)
    r70 = k70 * (HO2 * CH3 - OH * CH3O)
    r71 = k71 * (HO2 * CO - OH * CO2)
    r72 = k72 * (HO2 * CH2O - HCO * H2O2)
    r73 = k73 * (CH2 * O2 - OH * HCO)
    r74 = k74 * (CH2 * H2 - H * CH3)
    r75 = k75 * (CH2**2 - H2 * C2H2)
    r76 = k76 * (CH2 * CH3 - H * C2H4)
    r77 = k77 * (CH2 * CH4 - CH3**2)
    r78 = k78 * (CH2_S * N2 - CH2 * N2)
    r79 = k79 * (CH2_S * AR - CH2 * AR)
    r80 = k80 * (CH2_S * O2 - H * OH * CO)
    r81 = k81 * (CH2_S * O2 - CO * H2O)
    r82 = k82 * (CH2_S * H2 - CH3 * H)
    r83 = k83 * (CH2_S * H2O - CH2 * H2O)
    r84 = k84 * (CH2_S * CH3 - H * C2H4)
    r85 = k85 * (CH2_S * CH4 - CH3**2)
    r86 = k86 * (CH2_S * CO - CH2 * CO)
    r87 = k87 * (CH2_S * CO2 - CH2 * CO2)
    r88 = k88 * (CH2_S * CO2 - CO * CH2O)
    r89 = k89 * (CH3 * O2 - O * CH3O)
    r90 = k90 * (CH3 * O2 - OH * CH2O)
    r91 = k91 * (CH3 * H2O2 - HO2 * CH4)
    r92 = k92 * (CH3**2 - C2H6) * M92
    r93 = k93 * (CH3**2 - H * C2H5)
    r94 = k94 * (CH3 * HCO - CH4 * CO)
    r95 = k95 * (CH3 * CH2O - HCO * CH4)
    r96 = k96 * (CH3 * C2H4 - C2H3 * CH4)
    r97 = k97 * (CH3 * C2H6 - C2H5 * CH4)
    r98 = k98 * (HCO * H2O - H * CO * H2O)
    r99 = k99 * (HCO - H * CO) * M99
    r100 = k100 * (HCO * O2 - HO2 * CO)
    r101 = k101 * (CH3O * O2 - HO2 * CH2O)
    r102 = k102 * (C2H3 * O2 - HCO * CH2O)
    r103 = k103 * (C2H4 - H2 * C2H2) * M103
    r104 = k104 * (C2H5 * O2 - HO2 * C2H4)
    r105 = k105 * (N * NO - N2 * O)
    r106 = k106 * (N * O2 - NO * O)
    r107 = k107 * (N * OH - NO * H)
    r108 = k108 * (N2O * O - N2 * O2)
    r109 = k109 * (N2O * O - NO**2)
    r110 = k110 * (N2O * H - N2 * OH)
    r111 = k111 * (N2O * OH - N2 * HO2)
    r112 = k112 * (N2O - N2 * O) * M112
    r113 = k113 * (HO2 * NO - NO2 * OH)
    r114 = k114 * (NO * O - NO2) * M114
    r115 = k115 * (NO2 * O - NO * O2)
    r116 = k116 * (NO2 * H - NO * OH)

    # Species balances
    dH2 = - r2 + r25 + r26 + r27 + r28 + r30 + r32 + r35 + r37 + r39 + r43 + r45 + r47 - r48 - r49 - r74 + r75 \
        - r82 + r103
    dH = - r1 + r2 + r4 + r5 + r6 + r10 - r19 - r20 - r21 - r22 - r23 - r24 - 2*r25 - 2*r26 - 2*r27 - 2*r28 - r29 \
        - r30 - r31 - r32 - r33 - r34 - r35 - r36 - r37 - r38 - r39 - r40 - r41 - r42 - r43 - r44 - r45 - r46 \
        - r47 + r49 + r54 + r55 + r59 + r74 + r76 + r80 + r82 + r84 + r93 + r98 + r99 + r107 - r110 - r116
    dO = - r1 - r2 - r3 - r4 - r5 - r6 - r7 - r8 - r9 - r10 - r11 - r12 - r13 - r14 - r15 - r16 + r17 + r24 + r51 \
        + r89 + r105 + r106 - r108 - r109 + r112 - r114 - r115
    dO2 = r3 - r17 - r18 - r19 - r20 - r21 - r22 - r23 - r24 + r30 + r52 + r66 + r67 + r69 - r73 - r80 - r81 - r89 \
        - r90 - r100 - r101 - r102 - r104 - r106 + r108 + r115
    dOH = r1 + r2 + r3 + r7 + r9 + r11 + r16 + r24 - r29 + 2*r31 + r40 - r49 - 2*r50 - 2*r51 - r52 - r53 - r54 \
        - r55 - r56 - r57 - r58 - r59 - r60 - r61 - r62 - r63 - r64 - r65 + r68 + r70 + r71 + r73 + r80 + r90 \
        - r107 + r110 - r111 + r113 + r116
    dH2O = r29 + r49 + r51 + r52 + r53 + r56 + r57 + r58 + r60 + r61 + r63 + r64 + r65 + r81
    dHO2 = - r3 + r18 + r19 + r20 + r21 + r22 + r23 - r30 - r31 + r32 - r52 + r53 - 2*r66 - 2*r67 - r68 - r69 - r70 \
        - r71 - r72 + r91 + r100 + r101 + r104 + r111 - r113
    dH2O2 = - r32 + r50 - r53 + r66 + r67 + r72 - r91
    dCH2 = - r4 + r13 - r33 - r54 + r56 - r68 - r73 - r74 - 2*r75 - r76 - r77 + r78 + r79 + r83 + r86 + r87
    dCH2_S = - r5 + r12 - r55 + r57 - r78 - r79 - r80 - r81 - r82 - r83 - r84 - r85 - r86 - r87 - r88
    dCH3 = - r6 + r7 + r14 + r15 + r33 - r34 + r35 + r40 - r56 - r57 + r58 + r62 - r69 - r70 + r74 - r76 + 2*r77 \
        + r82 - r84 + 2*r85 - r89 - r90 - r91 - 2*r92 - 2*r93 - r94 - r95 - r96 - r97
    dCH4 = - r7 + r34 - r35 - r58 + r69 - r77 - r85 + r91 + r94 + r95 + r96 + r97
    dCO = - r8 + r9 + r12 + r13 - r17 + r37 - r48 - r59 + r60 + r62 - r71 + r80 + r81 + r88 + r94 + r98 + r99 \
        + r100
    dCO2 = r8 + r10 + r17 + r59 + r71 - r88
    dHCO = r4 + r5 - r9 - r10 + r11 + r14 + r18 - r36 - r37 + r39 - r60 + r61 + r72 + r73 - r94 + r95 - r98 - r99 \
        - r100 + r102
    dCH2O = r6 - r11 + r15 - r18 + r36 - r38 - r39 + r48 + r54 + r55 - r61 + r68 - r72 + r88 + r90 - r95 + r101 \
        + r102
    dCH3O = r38 - r40 + r70 + r89 - r101
    dC2H2 = - r12 - r13 - r41 + r43 - r62 + r63 + r75 + r103
    dC2H3 = r41 - r42 - r43 + r45 - r63 + r64 + r96 - r102
    dC2H4 = - r14 + r42 - r44 - r45 - r64 + r76 + r84 - r96 - r103 + r104
    dC2H5 = - r15 + r16 + r44 - r46 + r47 + r65 + r93 + r97 - r104
    dC2H6 = - r16 + r46 - r47 - r65 + r92 - r97
    dN2 = r105 + r108 + r110 + r111 + r112
    dAR = 0.0
    dNO = - r105 + r106 + r107 + 2*r109 - r113 - r114 + r115 + r116
    dNO2 = r113 + r114 - r115 - r116
    dN2O = - r108 - r109 - r110 - r111 - r112
    dN = - r105 - r106 - r107

    return np.array([dH2, dH, dO, dO2, dOH, dH2O, dHO2, dH2O2, dCH2, dCH2_S, dCH3, dCH4, dCO, dCO2, dHCO, dCH2O, dCH3O, dC2H2, dC2H3, dC2H4, dC2H5, dC2H6, dN2, dAR, dNO, dNO2, dN2O, dN], dtype=float)


def Smooke_ReducedModel(t: float, y: np.ndarray, k: np.ndarray, dim=False) -> np.ndarray:
    """
    Reduced 16-species / 35-reaction methane mechanism (Smooke).
    The stoichiometry is the published one, while k[i] are effective reaction-rate coefficients
    for this compact API rather than the original Arrhenius triplets.

    Source: https://raw.githubusercontent.com/jiweiqi/CollectionOfMechanisms/master/CH4_Methane/CH4_Smooke_s16r35/CH4_Smooke_s16r35.yaml
    Suggested observed species and external controls are returned in dim=True mode.
    """
    if dim == True:
        states = 16
        parameters = 35
        names = ['CH4', 'H2', 'O2', 'O', 'H', 'OH', 'HO2', 'H2O2', 'H2O', 'CO', 'CH3', 'CH2O', 'HCO', 'CH3O', 'CO2', 'N2']
        observed = ['CH4', 'O2', 'CO', 'CO2', 'H2O', 'OH', 'CH2O']
        inputs = ['feed_CH4', 'feed_O2', 'feed_N2', 'feed_H2O', 'Tin', 'pressure', 'residence_time', 'dilution']
        source = 'https://raw.githubusercontent.com/jiweiqi/CollectionOfMechanisms/master/CH4_Methane/CH4_Smooke_s16r35/CH4_Smooke_s16r35.yaml'
        return states, parameters, names, observed, inputs, source

    # Unpack species
    (
        CH4,
        H2,
        O2,
        O,
        H,
        OH,
        HO2,
        H2O2,
        H2O,
        CO,
        CH3,
        CH2O,
        HCO,
        CH3O,
        CO2,
        N2
    ) = y

    # Unpack effective reaction coefficients
    (
        k1,
        k2,
        k3,
        k4,
        k5,
        k6,
        k7,
        k8,
        k9,
        k10,
        k11,
        k12,
        k13,
        k14,
        k15,
        k16,
        k17,
        k18,
        k19,
        k20,
        k21,
        k22,
        k23,
        k24,
        k25,
        k26,
        k27,
        k28,
        k29,
        k30,
        k31,
        k32,
        k33,
        k34,
        k35
    ) = k

    # Third-body / falloff effective mixture concentrations
    M9 = 6.5*CH4 + H2 + 0.4*O2 + O + H + OH + HO2 + H2O2 + 6.5*H2O + 0.75*CO + CH3 + CH2O + HCO + CH3O + 1.5*CO2 + 0.4*N2
    M25 = 6.5*CH4 + H2 + 0.4*O2 + O + H + OH + HO2 + H2O2 + 6.5*H2O + 0.75*CO + CH3 + CH2O + HCO + CH3O + 1.5*CO2 + 0.4*N2
    M28 = 6.5*CH4 + H2 + 0.4*O2 + O + H + OH + HO2 + H2O2 + 6.5*H2O + 0.75*CO + CH3 + CH2O + HCO + CH3O + 1.5*CO2 + 0.4*N2
    M30 = 6.5*CH4 + H2 + 0.4*O2 + O + H + OH + HO2 + H2O2 + 6.5*H2O + 0.75*CO + CH3 + CH2O + HCO + CH3O + 1.5*CO2 + 0.4*N2
    M31 = 6.5*CH4 + H2 + 0.4*O2 + O + H + OH + HO2 + H2O2 + 6.5*H2O + 0.75*CO + CH3 + CH2O + HCO + CH3O + 1.5*CO2 + 0.4*N2
    M34 = 6.5*CH4 + H2 + 0.4*O2 + O + H + OH + HO2 + H2O2 + 6.5*H2O + 0.75*CO + CH3 + CH2O + HCO + CH3O + 1.5*CO2 + 0.4*N2
    M35 = 6.5*CH4 + H2 + 0.4*O2 + O + H + OH + HO2 + H2O2 + 6.5*H2O + 0.75*CO + CH3 + CH2O + HCO + CH3O + 1.5*CO2 + 0.4*N2

    # Reaction rates
    r1 = k1 * (H * O2)
    r2 = k2 * (O * OH)
    r3 = k3 * (O * H2)
    r4 = k4 * (OH * H)
    r5 = k5 * (H2 * OH)
    r6 = k6 * (H2O * H)
    r7 = k7 * (OH**2)
    r8 = k8 * (H2O * O)
    r9 = k9 * (H * O2) * M9
    r10 = k10 * (H * HO2)
    r11 = k11 * (H * HO2)
    r12 = k12 * (OH * HO2)
    r13 = k13 * (CO * OH)
    r14 = k14 * (CO2 * H)
    r15 = k15 * (CH4)
    r16 = k16 * (CH3 * H)
    r17 = k17 * (CH4 * H)
    r18 = k18 * (CH3 * H2)
    r19 = k19 * (CH4 * OH)
    r20 = k20 * (CH3 * H2O)
    r21 = k21 * (CH3 * O)
    r22 = k22 * (CH2O * H)
    r23 = k23 * (CH2O * OH)
    r24 = k24 * (HCO * H)
    r25 = k25 * (HCO) * M25
    r26 = k26 * (CH3 * O2)
    r27 = k27 * (CH3O * H)
    r28 = k28 * (CH3O) * M28
    r29 = k29 * (HO2**2)
    r30 = k30 * (H2O2) * M30
    r31 = k31 * (OH**2) * M31
    r32 = k32 * (H2O2 * OH)
    r33 = k33 * (H2O * HO2)
    r34 = k34 * (OH * H) * M34
    r35 = k35 * (H**2) * M35

    # Species balances
    dCH4 = - r15 + r16 - r17 + r18 - r19 + r20
    dH2 = - r3 + r4 - r5 + r6 + r11 + r17 - r18 + r22 + r24 + r27 + r35
    dO2 = - r1 + r2 - r9 + r11 + r12 - r26 + r29
    dO = r1 - r2 - r3 + r4 + r7 - r8 - r21 + r26
    dH = - r1 + r2 + r3 - r4 + r5 - r6 - r9 - r10 - r11 + r13 - r14 + r15 - r16 - r17 + r18 + r21 - r22 - r24 \
        + r25 - r27 + r28 - r34 - 2*r35
    dOH = r1 - r2 + r3 - r4 - r5 + r6 - 2*r7 + 2*r8 + 2*r10 - r12 - r13 + r14 - r19 + r20 - r23 + 2*r30 - 2*r31 \
        - r32 + r33 - r34
    dHO2 = r9 - r10 - r11 - r12 - 2*r29 + r32 - r33
    dH2O2 = r29 - r30 + r31 - r32 + r33
    dH2O = r5 - r6 + r7 - r8 + r12 + r19 - r20 + r23 + r32 - r33 + r34
    dCO = - r13 + r14 + r24 + r25
    dCH3 = r15 - r16 + r17 - r18 + r19 - r20 - r21 - r26
    dCH2O = r21 - r22 - r23 + r27 + r28
    dHCO = r22 + r23 - r24 - r25
    dCH3O = r26 - r27 - r28
    dCO2 = r13 - r14
    dN2 = 0.0

    return np.array([dCH4, dH2, dO2, dO, dH, dOH, dHO2, dH2O2, dH2O, dCO, dCH3, dCH2O, dHCO, dCH3O, dCO2, dN2], dtype=float)
