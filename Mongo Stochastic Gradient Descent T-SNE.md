

    from imp import reload
    import rsm
    import numpy as np
    import theano as T
    from matplotlib import pyplot as plt
    import utils
    import batch_data
    import time
    from batch_data import BatchData as Batch
    from tsne import bh_sne
    %matplotlib inline


    # if you need to reload the code shift-enter here !
    reload(utils)
    reload(batch_data)
    from batch_data import BatchData as Batch

Connect to **MongoDB**:


    # make sure mongo is running somewhere :
    utils.connect_to_database(database_name = 'yelp')

Use database to construct a lexicon (hash-table mapping words to vector
dimensions):


    lexicon, reverse_lexicon = utils.gather_lexicon('restaurants')

Create a **Replicated Softmax Machine**:


    # if you need to reload the replicated softmax code:
    reload(rsm);


    batch_size = 100
    learning_rate = 0.001 / batch_size
    encoder = rsm.RSM(momentum = 0.01, data = np.zeros([0,len(lexicon.items())]), hidden_units = 200, learning_rate=learning_rate)
    errors = np.zeros(0)

Create the stochastic batch element with 100 elements per mini-batch:


    rc = utils.ResourceConverter(lexicon = lexicon)
    batch = Batch(
        data=utils.mongo_database_global['restaurants'].find(), # from Mongo's cursor enumerator
        batch_size = batch_size,  # mini-batch
        shuffle = True, # stochastic
        conversion = rc.process # convert to matrices using lexicon)
    )

Start mini-batch learning for 1000 epochs:


    epochs = 20000
    batch.batch_size = 100
    new_errors = np.zeros(epochs)
    start_time = time.time()
    encoder.learning_rate = 0.001
    
    for epoch in range(epochs):
        if epoch > 0 and epoch % 200 == 0:
            encoder.k = int(1 + (float(epoch) / float(epochs)) * 4)
        if epoch > 0 and epoch % 300 == 0:
            encoder.learning_rate = max(0.00001, encoder.learning_rate.get_value()*0.999)
        encoder.data = batch.next()
        new_errors[epoch] = encoder.train()
        if epoch > 0 and epoch % 500 == 0:
            encoder.save("backup.pkz")
        if epoch > 0 and epoch % 10 == 0:
            print("Epoch[%2d] : PPL = %.02f [# Gibbs steps=%d] elapsed = %.05fmn" % (epoch, new_errors[epoch],encoder.k, (time.time() - start_time)/60.0))
    errors = np.append(errors, new_errors)

    Epoch[10] : PPL = 1460.82 [# Gibbs steps=1] elapsed = 0.11575mn
    Epoch[20] : PPL = 1631.27 [# Gibbs steps=1] elapsed = 0.21393mn
    Epoch[30] : PPL = 1602.98 [# Gibbs steps=1] elapsed = 0.30878mn
    Epoch[40] : PPL = 911.89 [# Gibbs steps=1] elapsed = 0.40446mn
    Epoch[50] : PPL = 1386.92 [# Gibbs steps=1] elapsed = 0.50318mn
    Epoch[60] : PPL = 1178.77 [# Gibbs steps=1] elapsed = 0.60920mn
    Epoch[70] : PPL = 1308.18 [# Gibbs steps=1] elapsed = 0.71057mn
    Epoch[80] : PPL = 1368.88 [# Gibbs steps=1] elapsed = 0.81835mn
    Epoch[90] : PPL = 1506.02 [# Gibbs steps=1] elapsed = 0.91462mn
    Epoch[100] : PPL = 1419.10 [# Gibbs steps=1] elapsed = 1.01687mn
    Epoch[110] : PPL = 1571.65 [# Gibbs steps=1] elapsed = 1.11725mn
    Epoch[120] : PPL = 1390.34 [# Gibbs steps=1] elapsed = 1.21412mn
    Epoch[130] : PPL = 1397.40 [# Gibbs steps=1] elapsed = 1.31094mn
    Epoch[140] : PPL = 1503.49 [# Gibbs steps=1] elapsed = 1.40861mn
    Epoch[150] : PPL = 1528.04 [# Gibbs steps=1] elapsed = 1.51002mn
    Epoch[160] : PPL = 1487.41 [# Gibbs steps=1] elapsed = 1.61126mn
    Epoch[170] : PPL = 926.19 [# Gibbs steps=1] elapsed = 1.70820mn
    Epoch[180] : PPL = 1527.05 [# Gibbs steps=1] elapsed = 1.81023mn
    Epoch[190] : PPL = 1327.07 [# Gibbs steps=1] elapsed = 1.91009mn
    Epoch[200] : PPL = 1375.66 [# Gibbs steps=1] elapsed = 2.01139mn
    Epoch[210] : PPL = 1324.14 [# Gibbs steps=1] elapsed = 2.11149mn
    Epoch[220] : PPL = 1577.29 [# Gibbs steps=1] elapsed = 2.20847mn
    Epoch[230] : PPL = 1534.30 [# Gibbs steps=1] elapsed = 2.30732mn
    Epoch[240] : PPL = 1216.86 [# Gibbs steps=1] elapsed = 2.40291mn
    Epoch[250] : PPL = 1626.71 [# Gibbs steps=1] elapsed = 2.50167mn
    Epoch[260] : PPL = 1289.79 [# Gibbs steps=1] elapsed = 2.60179mn
    Epoch[270] : PPL = 1173.79 [# Gibbs steps=1] elapsed = 2.69755mn
    Epoch[280] : PPL = 1450.38 [# Gibbs steps=1] elapsed = 2.79923mn
    Epoch[290] : PPL = 1130.46 [# Gibbs steps=1] elapsed = 2.89802mn
    Epoch[300] : PPL = 1453.00 [# Gibbs steps=1] elapsed = 2.99364mn
    Epoch[310] : PPL = 1583.75 [# Gibbs steps=1] elapsed = 3.09150mn
    Epoch[320] : PPL = 1492.83 [# Gibbs steps=1] elapsed = 3.19001mn
    Epoch[330] : PPL = 1537.39 [# Gibbs steps=1] elapsed = 3.28876mn
    Epoch[340] : PPL = 1134.34 [# Gibbs steps=1] elapsed = 3.38215mn
    Epoch[350] : PPL = 1419.68 [# Gibbs steps=1] elapsed = 3.47842mn
    Epoch[360] : PPL = 1510.29 [# Gibbs steps=1] elapsed = 3.57532mn
    Epoch[370] : PPL = 1654.85 [# Gibbs steps=1] elapsed = 3.67417mn
    Epoch[380] : PPL = 2003.06 [# Gibbs steps=1] elapsed = 3.77572mn
    Epoch[390] : PPL = 1420.25 [# Gibbs steps=1] elapsed = 3.87497mn
    Epoch[400] : PPL = 1456.03 [# Gibbs steps=1] elapsed = 3.97279mn
    Epoch[410] : PPL = 1431.77 [# Gibbs steps=1] elapsed = 4.06801mn
    Epoch[420] : PPL = 1975.01 [# Gibbs steps=1] elapsed = 4.16522mn
    Epoch[430] : PPL = 1296.67 [# Gibbs steps=1] elapsed = 4.26321mn
    Epoch[440] : PPL = 1383.96 [# Gibbs steps=1] elapsed = 4.36082mn
    Epoch[450] : PPL = 1313.24 [# Gibbs steps=1] elapsed = 4.46028mn
    Epoch[460] : PPL = 1430.68 [# Gibbs steps=1] elapsed = 4.55837mn
    Epoch[470] : PPL = 1625.63 [# Gibbs steps=1] elapsed = 4.65855mn
    Epoch[480] : PPL = 1123.09 [# Gibbs steps=1] elapsed = 4.75637mn
    Epoch[490] : PPL = 1176.25 [# Gibbs steps=1] elapsed = 4.85100mn
    Epoch[500] : PPL = 1522.51 [# Gibbs steps=1] elapsed = 5.12442mn
    Epoch[510] : PPL = 902.31 [# Gibbs steps=1] elapsed = 5.21852mn
    Epoch[520] : PPL = 1207.19 [# Gibbs steps=1] elapsed = 5.31715mn
    Epoch[530] : PPL = 1548.69 [# Gibbs steps=1] elapsed = 5.41564mn
    Epoch[540] : PPL = 1587.89 [# Gibbs steps=1] elapsed = 5.51569mn
    Epoch[550] : PPL = 1666.96 [# Gibbs steps=1] elapsed = 5.61494mn
    Epoch[560] : PPL = 1162.08 [# Gibbs steps=1] elapsed = 5.71398mn
    Epoch[570] : PPL = 858.67 [# Gibbs steps=1] elapsed = 5.80687mn
    Epoch[580] : PPL = 1185.40 [# Gibbs steps=1] elapsed = 5.90423mn
    Epoch[590] : PPL = 1536.06 [# Gibbs steps=1] elapsed = 5.99855mn
    Epoch[600] : PPL = 1314.58 [# Gibbs steps=1] elapsed = 6.09651mn
    Epoch[610] : PPL = 1522.12 [# Gibbs steps=1] elapsed = 6.19591mn
    Epoch[620] : PPL = 1645.64 [# Gibbs steps=1] elapsed = 6.29391mn
    Epoch[630] : PPL = 1573.87 [# Gibbs steps=1] elapsed = 6.39292mn
    Epoch[640] : PPL = 1044.57 [# Gibbs steps=1] elapsed = 6.49118mn
    Epoch[650] : PPL = 1298.02 [# Gibbs steps=1] elapsed = 6.59257mn
    Epoch[660] : PPL = 1495.62 [# Gibbs steps=1] elapsed = 6.68940mn
    Epoch[670] : PPL = 1107.73 [# Gibbs steps=1] elapsed = 6.78786mn
    Epoch[680] : PPL = 1519.51 [# Gibbs steps=1] elapsed = 6.88818mn
    Epoch[690] : PPL = 1557.16 [# Gibbs steps=1] elapsed = 6.98721mn
    Epoch[700] : PPL = 1446.36 [# Gibbs steps=1] elapsed = 7.08248mn
    Epoch[710] : PPL = 1337.64 [# Gibbs steps=1] elapsed = 7.18092mn
    Epoch[720] : PPL = 1485.16 [# Gibbs steps=1] elapsed = 7.28176mn
    Epoch[730] : PPL = 1946.74 [# Gibbs steps=1] elapsed = 7.38001mn
    Epoch[740] : PPL = 1376.23 [# Gibbs steps=1] elapsed = 7.47713mn
    Epoch[750] : PPL = 1483.76 [# Gibbs steps=1] elapsed = 7.57782mn
    Epoch[760] : PPL = 1669.69 [# Gibbs steps=1] elapsed = 7.67995mn
    Epoch[770] : PPL = 1588.16 [# Gibbs steps=1] elapsed = 7.77792mn
    Epoch[780] : PPL = 1394.06 [# Gibbs steps=1] elapsed = 7.87840mn
    Epoch[790] : PPL = 1038.70 [# Gibbs steps=1] elapsed = 7.97047mn
    Epoch[800] : PPL = 1394.12 [# Gibbs steps=1] elapsed = 8.06965mn
    Epoch[810] : PPL = 1473.85 [# Gibbs steps=1] elapsed = 8.16836mn
    Epoch[820] : PPL = 1327.72 [# Gibbs steps=1] elapsed = 8.26846mn
    Epoch[830] : PPL = 1088.71 [# Gibbs steps=1] elapsed = 8.36513mn
    Epoch[840] : PPL = 1636.69 [# Gibbs steps=1] elapsed = 8.46701mn
    Epoch[850] : PPL = 1181.64 [# Gibbs steps=1] elapsed = 8.56653mn
    Epoch[860] : PPL = 1525.81 [# Gibbs steps=1] elapsed = 8.66664mn
    Epoch[870] : PPL = 1673.08 [# Gibbs steps=1] elapsed = 8.76530mn
    Epoch[880] : PPL = 1519.51 [# Gibbs steps=1] elapsed = 8.86351mn
    Epoch[890] : PPL = 1301.93 [# Gibbs steps=1] elapsed = 8.96355mn
    Epoch[900] : PPL = 1378.71 [# Gibbs steps=1] elapsed = 9.06288mn
    Epoch[910] : PPL = 935.14 [# Gibbs steps=1] elapsed = 9.15766mn
    Epoch[920] : PPL = 1290.48 [# Gibbs steps=1] elapsed = 9.25689mn
    Epoch[930] : PPL = 1158.97 [# Gibbs steps=1] elapsed = 9.35284mn
    Epoch[940] : PPL = 1017.65 [# Gibbs steps=1] elapsed = 9.45055mn
    Epoch[950] : PPL = 1309.55 [# Gibbs steps=1] elapsed = 9.54617mn
    Epoch[960] : PPL = 1871.90 [# Gibbs steps=1] elapsed = 9.64659mn
    Epoch[970] : PPL = 1018.89 [# Gibbs steps=1] elapsed = 9.74326mn
    Epoch[980] : PPL = 1303.66 [# Gibbs steps=1] elapsed = 9.84365mn
    Epoch[990] : PPL = 890.86 [# Gibbs steps=1] elapsed = 9.94014mn
    Epoch[1000] : PPL = 1356.72 [# Gibbs steps=1] elapsed = 10.21406mn
    Epoch[1010] : PPL = 1571.14 [# Gibbs steps=1] elapsed = 10.31435mn
    Epoch[1020] : PPL = 1222.64 [# Gibbs steps=1] elapsed = 10.41359mn
    Epoch[1030] : PPL = 1223.42 [# Gibbs steps=1] elapsed = 10.50918mn
    Epoch[1040] : PPL = 1315.97 [# Gibbs steps=1] elapsed = 10.60996mn
    Epoch[1050] : PPL = 1460.74 [# Gibbs steps=1] elapsed = 10.70605mn
    Epoch[1060] : PPL = 1693.09 [# Gibbs steps=1] elapsed = 10.80496mn
    Epoch[1070] : PPL = 1291.58 [# Gibbs steps=1] elapsed = 10.90521mn
    Epoch[1080] : PPL = 1456.00 [# Gibbs steps=1] elapsed = 11.00565mn
    Epoch[1090] : PPL = 1528.92 [# Gibbs steps=1] elapsed = 11.10416mn
    Epoch[1100] : PPL = 1511.93 [# Gibbs steps=1] elapsed = 11.20244mn
    Epoch[1110] : PPL = 1371.44 [# Gibbs steps=1] elapsed = 11.30061mn
    Epoch[1120] : PPL = 1444.03 [# Gibbs steps=1] elapsed = 11.39978mn
    Epoch[1130] : PPL = 886.78 [# Gibbs steps=1] elapsed = 11.49647mn
    Epoch[1140] : PPL = 1525.31 [# Gibbs steps=1] elapsed = 11.59354mn
    Epoch[1150] : PPL = 1399.95 [# Gibbs steps=1] elapsed = 11.69124mn
    Epoch[1160] : PPL = 891.48 [# Gibbs steps=1] elapsed = 11.79013mn
    Epoch[1170] : PPL = 1316.11 [# Gibbs steps=1] elapsed = 11.88584mn
    Epoch[1180] : PPL = 1464.74 [# Gibbs steps=1] elapsed = 11.98577mn
    Epoch[1190] : PPL = 1120.26 [# Gibbs steps=1] elapsed = 12.08444mn
    Epoch[1200] : PPL = 1063.31 [# Gibbs steps=1] elapsed = 12.18147mn
    Epoch[1210] : PPL = 1535.10 [# Gibbs steps=1] elapsed = 12.27896mn
    Epoch[1220] : PPL = 1338.14 [# Gibbs steps=1] elapsed = 12.37702mn
    Epoch[1230] : PPL = 1076.70 [# Gibbs steps=1] elapsed = 12.47434mn
    Epoch[1240] : PPL = 1629.58 [# Gibbs steps=1] elapsed = 12.57248mn
    Epoch[1250] : PPL = 1350.16 [# Gibbs steps=1] elapsed = 12.66796mn
    Epoch[1260] : PPL = 1144.93 [# Gibbs steps=1] elapsed = 12.76697mn
    Epoch[1270] : PPL = 1311.72 [# Gibbs steps=1] elapsed = 12.86568mn
    Epoch[1280] : PPL = 1115.38 [# Gibbs steps=1] elapsed = 12.96181mn
    Epoch[1290] : PPL = 858.04 [# Gibbs steps=1] elapsed = 13.05825mn
    Epoch[1300] : PPL = 1275.03 [# Gibbs steps=1] elapsed = 13.15831mn
    Epoch[1310] : PPL = 1314.85 [# Gibbs steps=1] elapsed = 13.25733mn
    Epoch[1320] : PPL = 1383.81 [# Gibbs steps=1] elapsed = 13.35476mn
    Epoch[1330] : PPL = 907.18 [# Gibbs steps=1] elapsed = 13.45109mn
    Epoch[1340] : PPL = 1255.90 [# Gibbs steps=1] elapsed = 13.54806mn
    Epoch[1350] : PPL = 1652.40 [# Gibbs steps=1] elapsed = 13.64399mn
    Epoch[1360] : PPL = 1435.64 [# Gibbs steps=1] elapsed = 13.74282mn
    Epoch[1370] : PPL = 1192.47 [# Gibbs steps=1] elapsed = 13.84021mn
    Epoch[1380] : PPL = 1504.00 [# Gibbs steps=1] elapsed = 13.93879mn
    Epoch[1390] : PPL = 1512.83 [# Gibbs steps=1] elapsed = 14.03637mn
    Epoch[1400] : PPL = 1502.83 [# Gibbs steps=1] elapsed = 14.13782mn
    Epoch[1410] : PPL = 1571.42 [# Gibbs steps=1] elapsed = 14.23843mn
    Epoch[1420] : PPL = 936.88 [# Gibbs steps=1] elapsed = 14.33380mn
    Epoch[1430] : PPL = 1126.97 [# Gibbs steps=1] elapsed = 14.43157mn
    Epoch[1440] : PPL = 1383.74 [# Gibbs steps=1] elapsed = 14.52965mn
    Epoch[1450] : PPL = 1249.88 [# Gibbs steps=1] elapsed = 14.62995mn
    Epoch[1460] : PPL = 955.75 [# Gibbs steps=1] elapsed = 14.72848mn
    Epoch[1470] : PPL = 1736.95 [# Gibbs steps=1] elapsed = 14.82948mn
    Epoch[1480] : PPL = 1487.67 [# Gibbs steps=1] elapsed = 14.93041mn
    Epoch[1490] : PPL = 1027.15 [# Gibbs steps=1] elapsed = 15.02817mn
    Epoch[1500] : PPL = 1273.23 [# Gibbs steps=1] elapsed = 15.29987mn
    Epoch[1510] : PPL = 1316.10 [# Gibbs steps=1] elapsed = 15.39872mn
    Epoch[1520] : PPL = 893.99 [# Gibbs steps=1] elapsed = 15.49500mn
    Epoch[1530] : PPL = 859.07 [# Gibbs steps=1] elapsed = 15.59218mn
    Epoch[1540] : PPL = 1562.20 [# Gibbs steps=1] elapsed = 15.69407mn
    Epoch[1550] : PPL = 1095.15 [# Gibbs steps=1] elapsed = 15.79184mn
    Epoch[1560] : PPL = 1371.61 [# Gibbs steps=1] elapsed = 15.89037mn
    Epoch[1570] : PPL = 1305.06 [# Gibbs steps=1] elapsed = 15.99144mn
    Epoch[1580] : PPL = 1482.59 [# Gibbs steps=1] elapsed = 16.08982mn
    Epoch[1590] : PPL = 1238.73 [# Gibbs steps=1] elapsed = 16.18351mn
    Epoch[1600] : PPL = 1323.05 [# Gibbs steps=1] elapsed = 16.28294mn
    Epoch[1610] : PPL = 1392.68 [# Gibbs steps=1] elapsed = 16.38179mn
    Epoch[1620] : PPL = 931.38 [# Gibbs steps=1] elapsed = 16.47944mn
    Epoch[1630] : PPL = 1325.40 [# Gibbs steps=1] elapsed = 16.57261mn
    Epoch[1640] : PPL = 1426.44 [# Gibbs steps=1] elapsed = 16.66929mn
    Epoch[1650] : PPL = 1514.56 [# Gibbs steps=1] elapsed = 16.76724mn
    Epoch[1660] : PPL = 1577.98 [# Gibbs steps=1] elapsed = 16.86714mn
    Epoch[1670] : PPL = 1399.30 [# Gibbs steps=1] elapsed = 16.96147mn
    Epoch[1680] : PPL = 1500.26 [# Gibbs steps=1] elapsed = 17.06112mn
    Epoch[1690] : PPL = 1011.46 [# Gibbs steps=1] elapsed = 17.15773mn
    Epoch[1700] : PPL = 1578.17 [# Gibbs steps=1] elapsed = 17.25753mn
    Epoch[1710] : PPL = 1523.96 [# Gibbs steps=1] elapsed = 17.35619mn
    Epoch[1720] : PPL = 1323.59 [# Gibbs steps=1] elapsed = 17.45339mn
    Epoch[1730] : PPL = 1530.80 [# Gibbs steps=1] elapsed = 17.55045mn
    Epoch[1740] : PPL = 886.23 [# Gibbs steps=1] elapsed = 17.64637mn
    Epoch[1750] : PPL = 1327.35 [# Gibbs steps=1] elapsed = 17.74487mn
    Epoch[1760] : PPL = 1634.60 [# Gibbs steps=1] elapsed = 17.84737mn
    Epoch[1770] : PPL = 1018.92 [# Gibbs steps=1] elapsed = 17.94126mn
    Epoch[1780] : PPL = 1334.58 [# Gibbs steps=1] elapsed = 18.03862mn
    Epoch[1790] : PPL = 1585.23 [# Gibbs steps=1] elapsed = 18.13839mn
    Epoch[1800] : PPL = 1532.88 [# Gibbs steps=1] elapsed = 18.23933mn
    Epoch[1810] : PPL = 889.67 [# Gibbs steps=1] elapsed = 18.33460mn
    Epoch[1820] : PPL = 948.87 [# Gibbs steps=1] elapsed = 18.42941mn
    Epoch[1830] : PPL = 1379.07 [# Gibbs steps=1] elapsed = 18.52980mn
    Epoch[1840] : PPL = 1153.21 [# Gibbs steps=1] elapsed = 18.62651mn
    Epoch[1850] : PPL = 1543.42 [# Gibbs steps=1] elapsed = 18.72317mn
    Epoch[1860] : PPL = 1528.53 [# Gibbs steps=1] elapsed = 18.82323mn
    Epoch[1870] : PPL = 981.87 [# Gibbs steps=1] elapsed = 18.92056mn
    Epoch[1880] : PPL = 1265.81 [# Gibbs steps=1] elapsed = 19.01636mn
    Epoch[1890] : PPL = 1623.08 [# Gibbs steps=1] elapsed = 19.11419mn
    Epoch[1900] : PPL = 1293.81 [# Gibbs steps=1] elapsed = 19.21366mn
    Epoch[1910] : PPL = 1941.39 [# Gibbs steps=1] elapsed = 19.31032mn
    Epoch[1920] : PPL = 1547.03 [# Gibbs steps=1] elapsed = 19.40723mn
    Epoch[1930] : PPL = 888.00 [# Gibbs steps=1] elapsed = 19.50555mn
    Epoch[1940] : PPL = 825.83 [# Gibbs steps=1] elapsed = 19.60035mn
    Epoch[1950] : PPL = 1313.60 [# Gibbs steps=1] elapsed = 19.70092mn
    Epoch[1960] : PPL = 1608.53 [# Gibbs steps=1] elapsed = 19.80138mn
    Epoch[1970] : PPL = 1397.34 [# Gibbs steps=1] elapsed = 19.90394mn
    Epoch[1980] : PPL = 1647.09 [# Gibbs steps=1] elapsed = 20.00208mn
    Epoch[1990] : PPL = 1112.59 [# Gibbs steps=1] elapsed = 20.10132mn
    Epoch[2000] : PPL = 1553.96 [# Gibbs steps=1] elapsed = 20.37538mn
    Epoch[2010] : PPL = 1320.88 [# Gibbs steps=1] elapsed = 20.47433mn
    Epoch[2020] : PPL = 1465.82 [# Gibbs steps=1] elapsed = 20.57402mn
    Epoch[2030] : PPL = 929.56 [# Gibbs steps=1] elapsed = 20.67190mn
    Epoch[2040] : PPL = 1535.33 [# Gibbs steps=1] elapsed = 20.77053mn
    Epoch[2050] : PPL = 1511.57 [# Gibbs steps=1] elapsed = 20.86886mn
    Epoch[2060] : PPL = 1063.55 [# Gibbs steps=1] elapsed = 20.96506mn
    Epoch[2070] : PPL = 1326.57 [# Gibbs steps=1] elapsed = 21.06346mn
    Epoch[2080] : PPL = 1403.83 [# Gibbs steps=1] elapsed = 21.16187mn
    Epoch[2090] : PPL = 1039.25 [# Gibbs steps=1] elapsed = 21.26127mn
    Epoch[2100] : PPL = 1298.67 [# Gibbs steps=1] elapsed = 21.35894mn
    Epoch[2110] : PPL = 1499.89 [# Gibbs steps=1] elapsed = 21.45400mn
    Epoch[2120] : PPL = 1622.50 [# Gibbs steps=1] elapsed = 21.55193mn
    Epoch[2130] : PPL = 1317.49 [# Gibbs steps=1] elapsed = 21.65130mn
    Epoch[2140] : PPL = 1286.03 [# Gibbs steps=1] elapsed = 21.75002mn
    Epoch[2150] : PPL = 1097.25 [# Gibbs steps=1] elapsed = 21.84474mn
    Epoch[2160] : PPL = 1588.37 [# Gibbs steps=1] elapsed = 21.94116mn
    Epoch[2170] : PPL = 1563.82 [# Gibbs steps=1] elapsed = 22.03875mn
    Epoch[2180] : PPL = 1372.12 [# Gibbs steps=1] elapsed = 22.14029mn
    Epoch[2190] : PPL = 1492.48 [# Gibbs steps=1] elapsed = 22.23764mn
    Epoch[2200] : PPL = 1126.97 [# Gibbs steps=1] elapsed = 22.33502mn
    Epoch[2210] : PPL = 1436.58 [# Gibbs steps=1] elapsed = 22.43576mn
    Epoch[2220] : PPL = 1468.77 [# Gibbs steps=1] elapsed = 22.53816mn
    Epoch[2230] : PPL = 1435.85 [# Gibbs steps=1] elapsed = 22.63284mn
    Epoch[2240] : PPL = 1429.96 [# Gibbs steps=1] elapsed = 22.73003mn
    Epoch[2250] : PPL = 1179.84 [# Gibbs steps=1] elapsed = 22.82704mn
    Epoch[2260] : PPL = 1571.59 [# Gibbs steps=1] elapsed = 22.92527mn
    Epoch[2270] : PPL = 972.40 [# Gibbs steps=1] elapsed = 23.02238mn
    Epoch[2280] : PPL = 1602.47 [# Gibbs steps=1] elapsed = 23.11969mn
    Epoch[2290] : PPL = 1372.14 [# Gibbs steps=1] elapsed = 23.21969mn
    Epoch[2300] : PPL = 1557.63 [# Gibbs steps=1] elapsed = 23.31779mn
    Epoch[2310] : PPL = 1321.17 [# Gibbs steps=1] elapsed = 23.41929mn
    Epoch[2320] : PPL = 1489.13 [# Gibbs steps=1] elapsed = 23.51647mn
    Epoch[2330] : PPL = 1119.45 [# Gibbs steps=1] elapsed = 23.61204mn
    Epoch[2340] : PPL = 1661.53 [# Gibbs steps=1] elapsed = 23.70886mn
    Epoch[2350] : PPL = 920.86 [# Gibbs steps=1] elapsed = 23.80709mn
    Epoch[2360] : PPL = 1437.55 [# Gibbs steps=1] elapsed = 23.90384mn
    Epoch[2370] : PPL = 1521.51 [# Gibbs steps=1] elapsed = 23.99971mn
    Epoch[2380] : PPL = 1306.65 [# Gibbs steps=1] elapsed = 24.09771mn
    Epoch[2390] : PPL = 1674.19 [# Gibbs steps=1] elapsed = 24.19572mn
    Epoch[2400] : PPL = 1567.96 [# Gibbs steps=1] elapsed = 24.29378mn
    Epoch[2410] : PPL = 1236.09 [# Gibbs steps=1] elapsed = 24.39123mn
    Epoch[2420] : PPL = 1224.99 [# Gibbs steps=1] elapsed = 24.49097mn
    Epoch[2430] : PPL = 1081.69 [# Gibbs steps=1] elapsed = 24.58766mn
    Epoch[2440] : PPL = 1541.32 [# Gibbs steps=1] elapsed = 24.68573mn
    Epoch[2450] : PPL = 1070.09 [# Gibbs steps=1] elapsed = 24.78160mn
    Epoch[2460] : PPL = 969.09 [# Gibbs steps=1] elapsed = 24.87925mn
    Epoch[2470] : PPL = 1535.27 [# Gibbs steps=1] elapsed = 24.97566mn
    Epoch[2480] : PPL = 1395.44 [# Gibbs steps=1] elapsed = 25.07523mn
    Epoch[2490] : PPL = 1505.48 [# Gibbs steps=1] elapsed = 25.17197mn
    Epoch[2500] : PPL = 1094.38 [# Gibbs steps=1] elapsed = 25.44857mn
    Epoch[2510] : PPL = 1486.93 [# Gibbs steps=1] elapsed = 25.54775mn
    Epoch[2520] : PPL = 1474.42 [# Gibbs steps=1] elapsed = 25.64605mn
    Epoch[2530] : PPL = 1147.04 [# Gibbs steps=1] elapsed = 25.74246mn
    Epoch[2540] : PPL = 853.01 [# Gibbs steps=1] elapsed = 25.83848mn
    Epoch[2550] : PPL = 1613.43 [# Gibbs steps=1] elapsed = 25.94004mn
    Epoch[2560] : PPL = 1341.98 [# Gibbs steps=1] elapsed = 26.03721mn
    Epoch[2570] : PPL = 1422.00 [# Gibbs steps=1] elapsed = 26.13657mn
    Epoch[2580] : PPL = 1638.79 [# Gibbs steps=1] elapsed = 26.23452mn
    Epoch[2590] : PPL = 876.52 [# Gibbs steps=1] elapsed = 26.32827mn
    Epoch[2600] : PPL = 1573.10 [# Gibbs steps=1] elapsed = 26.42585mn
    Epoch[2610] : PPL = 1245.62 [# Gibbs steps=1] elapsed = 26.52525mn
    Epoch[2620] : PPL = 1181.30 [# Gibbs steps=1] elapsed = 26.62275mn
    Epoch[2630] : PPL = 1500.80 [# Gibbs steps=1] elapsed = 26.72088mn
    Epoch[2640] : PPL = 1454.83 [# Gibbs steps=1] elapsed = 26.82028mn
    Epoch[2650] : PPL = 1376.28 [# Gibbs steps=1] elapsed = 26.91985mn
    Epoch[2660] : PPL = 1280.34 [# Gibbs steps=1] elapsed = 27.01663mn
    Epoch[2670] : PPL = 1263.80 [# Gibbs steps=1] elapsed = 27.11491mn
    Epoch[2680] : PPL = 1280.42 [# Gibbs steps=1] elapsed = 27.21302mn
    Epoch[2690] : PPL = 1360.42 [# Gibbs steps=1] elapsed = 27.31771mn
    Epoch[2700] : PPL = 1230.51 [# Gibbs steps=1] elapsed = 27.41287mn
    Epoch[2710] : PPL = 1529.32 [# Gibbs steps=1] elapsed = 27.51276mn
    Epoch[2720] : PPL = 1484.63 [# Gibbs steps=1] elapsed = 27.60720mn
    Epoch[2730] : PPL = 1651.45 [# Gibbs steps=1] elapsed = 27.70490mn
    Epoch[2740] : PPL = 1507.87 [# Gibbs steps=1] elapsed = 27.80461mn
    Epoch[2750] : PPL = 1229.45 [# Gibbs steps=1] elapsed = 27.90340mn
    Epoch[2760] : PPL = 1492.65 [# Gibbs steps=1] elapsed = 28.00736mn
    Epoch[2770] : PPL = 1054.31 [# Gibbs steps=1] elapsed = 28.10708mn
    Epoch[2780] : PPL = 1515.32 [# Gibbs steps=1] elapsed = 28.20569mn
    Epoch[2790] : PPL = 1507.36 [# Gibbs steps=1] elapsed = 28.30507mn
    Epoch[2800] : PPL = 1509.31 [# Gibbs steps=1] elapsed = 28.40068mn
    Epoch[2810] : PPL = 1052.54 [# Gibbs steps=1] elapsed = 28.49701mn
    Epoch[2820] : PPL = 1236.87 [# Gibbs steps=1] elapsed = 28.59304mn
    Epoch[2830] : PPL = 893.24 [# Gibbs steps=1] elapsed = 28.69201mn
    Epoch[2840] : PPL = 1370.85 [# Gibbs steps=1] elapsed = 28.79293mn
    Epoch[2850] : PPL = 1343.92 [# Gibbs steps=1] elapsed = 28.89014mn
    Epoch[2860] : PPL = 1601.83 [# Gibbs steps=1] elapsed = 28.98941mn
    Epoch[2870] : PPL = 1362.05 [# Gibbs steps=1] elapsed = 29.08969mn
    Epoch[2880] : PPL = 1000.66 [# Gibbs steps=1] elapsed = 29.18391mn
    Epoch[2890] : PPL = 1303.54 [# Gibbs steps=1] elapsed = 29.28569mn
    Epoch[2900] : PPL = 1446.66 [# Gibbs steps=1] elapsed = 29.38475mn
    Epoch[2910] : PPL = 1317.80 [# Gibbs steps=1] elapsed = 29.48114mn
    Epoch[2920] : PPL = 1475.01 [# Gibbs steps=1] elapsed = 29.58267mn
    Epoch[2930] : PPL = 1405.43 [# Gibbs steps=1] elapsed = 29.68138mn
    Epoch[2940] : PPL = 1047.43 [# Gibbs steps=1] elapsed = 29.77922mn
    Epoch[2950] : PPL = 883.75 [# Gibbs steps=1] elapsed = 29.87616mn
    Epoch[2960] : PPL = 846.01 [# Gibbs steps=1] elapsed = 29.97492mn
    Epoch[2970] : PPL = 1630.05 [# Gibbs steps=1] elapsed = 30.07605mn
    Epoch[2980] : PPL = 1430.08 [# Gibbs steps=1] elapsed = 30.17393mn
    Epoch[2990] : PPL = 1305.79 [# Gibbs steps=1] elapsed = 30.27305mn
    Epoch[3000] : PPL = 1095.16 [# Gibbs steps=1] elapsed = 30.54559mn
    Epoch[3010] : PPL = 1461.72 [# Gibbs steps=1] elapsed = 30.64690mn
    Epoch[3020] : PPL = 1139.29 [# Gibbs steps=1] elapsed = 30.74551mn
    Epoch[3030] : PPL = 1525.04 [# Gibbs steps=1] elapsed = 30.84387mn
    Epoch[3040] : PPL = 1514.16 [# Gibbs steps=1] elapsed = 30.94244mn
    Epoch[3050] : PPL = 1228.60 [# Gibbs steps=1] elapsed = 31.04111mn
    Epoch[3060] : PPL = 1482.58 [# Gibbs steps=1] elapsed = 31.14195mn
    Epoch[3070] : PPL = 1834.87 [# Gibbs steps=1] elapsed = 31.24051mn
    Epoch[3080] : PPL = 1534.48 [# Gibbs steps=1] elapsed = 31.34176mn
    Epoch[3090] : PPL = 1041.98 [# Gibbs steps=1] elapsed = 31.43804mn
    Epoch[3100] : PPL = 1016.50 [# Gibbs steps=1] elapsed = 31.53840mn
    Epoch[3110] : PPL = 1500.45 [# Gibbs steps=1] elapsed = 31.63767mn
    Epoch[3120] : PPL = 1439.35 [# Gibbs steps=1] elapsed = 31.73430mn
    Epoch[3130] : PPL = 871.96 [# Gibbs steps=1] elapsed = 31.82990mn
    Epoch[3140] : PPL = 1256.94 [# Gibbs steps=1] elapsed = 31.92568mn
    Epoch[3150] : PPL = 1432.19 [# Gibbs steps=1] elapsed = 32.02201mn
    Epoch[3160] : PPL = 1374.89 [# Gibbs steps=1] elapsed = 32.11892mn
    Epoch[3170] : PPL = 1445.62 [# Gibbs steps=1] elapsed = 32.21916mn
    Epoch[3180] : PPL = 1576.89 [# Gibbs steps=1] elapsed = 32.31609mn
    Epoch[3190] : PPL = 1810.07 [# Gibbs steps=1] elapsed = 32.41598mn
    Epoch[3200] : PPL = 1529.98 [# Gibbs steps=1] elapsed = 32.51310mn
    Epoch[3210] : PPL = 1268.52 [# Gibbs steps=1] elapsed = 32.60800mn
    Epoch[3220] : PPL = 1059.60 [# Gibbs steps=1] elapsed = 32.70599mn
    Epoch[3230] : PPL = 1438.03 [# Gibbs steps=1] elapsed = 32.80036mn
    Epoch[3240] : PPL = 1390.15 [# Gibbs steps=1] elapsed = 32.89854mn
    Epoch[3250] : PPL = 1078.65 [# Gibbs steps=1] elapsed = 32.99767mn
    Epoch[3260] : PPL = 1257.49 [# Gibbs steps=1] elapsed = 33.09736mn
    Epoch[3270] : PPL = 1569.55 [# Gibbs steps=1] elapsed = 33.19169mn
    Epoch[3280] : PPL = 1321.21 [# Gibbs steps=1] elapsed = 33.29133mn
    Epoch[3290] : PPL = 1303.88 [# Gibbs steps=1] elapsed = 33.38901mn
    Epoch[3300] : PPL = 1483.44 [# Gibbs steps=1] elapsed = 33.48874mn
    Epoch[3310] : PPL = 1449.31 [# Gibbs steps=1] elapsed = 33.58767mn
    Epoch[3320] : PPL = 1519.89 [# Gibbs steps=1] elapsed = 33.68414mn
    Epoch[3330] : PPL = 1310.56 [# Gibbs steps=1] elapsed = 33.78241mn
    Epoch[3340] : PPL = 893.61 [# Gibbs steps=1] elapsed = 33.87516mn
    Epoch[3350] : PPL = 1092.56 [# Gibbs steps=1] elapsed = 33.97201mn
    Epoch[3360] : PPL = 1086.50 [# Gibbs steps=1] elapsed = 34.07054mn
    Epoch[3370] : PPL = 1398.09 [# Gibbs steps=1] elapsed = 34.16704mn
    Epoch[3380] : PPL = 1548.30 [# Gibbs steps=1] elapsed = 34.26515mn
    Epoch[3390] : PPL = 1057.84 [# Gibbs steps=1] elapsed = 34.36216mn
    Epoch[3400] : PPL = 1150.55 [# Gibbs steps=1] elapsed = 34.46022mn
    Epoch[3410] : PPL = 1559.52 [# Gibbs steps=1] elapsed = 34.55865mn
    Epoch[3420] : PPL = 1101.22 [# Gibbs steps=1] elapsed = 34.65697mn
    Epoch[3430] : PPL = 1425.86 [# Gibbs steps=1] elapsed = 34.75440mn
    Epoch[3440] : PPL = 907.20 [# Gibbs steps=1] elapsed = 34.85015mn
    Epoch[3450] : PPL = 1483.53 [# Gibbs steps=1] elapsed = 34.94994mn
    Epoch[3460] : PPL = 1463.77 [# Gibbs steps=1] elapsed = 35.04808mn
    Epoch[3470] : PPL = 1362.13 [# Gibbs steps=1] elapsed = 35.14586mn
    Epoch[3480] : PPL = 1447.17 [# Gibbs steps=1] elapsed = 35.24215mn
    Epoch[3490] : PPL = 1067.37 [# Gibbs steps=1] elapsed = 35.34001mn
    Epoch[3500] : PPL = 1457.49 [# Gibbs steps=1] elapsed = 35.61860mn
    Epoch[3510] : PPL = 1523.15 [# Gibbs steps=1] elapsed = 35.71640mn
    Epoch[3520] : PPL = 1307.85 [# Gibbs steps=1] elapsed = 35.81447mn
    Epoch[3530] : PPL = 1216.50 [# Gibbs steps=1] elapsed = 35.90874mn
    Epoch[3540] : PPL = 1257.37 [# Gibbs steps=1] elapsed = 36.01014mn
    Epoch[3550] : PPL = 1574.30 [# Gibbs steps=1] elapsed = 36.11029mn
    Epoch[3560] : PPL = 1431.09 [# Gibbs steps=1] elapsed = 36.20897mn
    Epoch[3570] : PPL = 1368.31 [# Gibbs steps=1] elapsed = 36.30756mn
    Epoch[3580] : PPL = 906.66 [# Gibbs steps=1] elapsed = 36.40461mn
    Epoch[3590] : PPL = 954.45 [# Gibbs steps=1] elapsed = 36.50237mn
    Epoch[3600] : PPL = 837.71 [# Gibbs steps=1] elapsed = 36.59887mn
    Epoch[3610] : PPL = 1594.14 [# Gibbs steps=1] elapsed = 36.69706mn
    Epoch[3620] : PPL = 1527.93 [# Gibbs steps=1] elapsed = 36.79595mn
    Epoch[3630] : PPL = 1441.59 [# Gibbs steps=1] elapsed = 36.89215mn
    Epoch[3640] : PPL = 1522.77 [# Gibbs steps=1] elapsed = 36.98968mn
    Epoch[3650] : PPL = 1226.25 [# Gibbs steps=1] elapsed = 37.08601mn
    Epoch[3660] : PPL = 1100.25 [# Gibbs steps=1] elapsed = 37.18189mn
    Epoch[3670] : PPL = 1362.44 [# Gibbs steps=1] elapsed = 37.27719mn
    Epoch[3680] : PPL = 1453.11 [# Gibbs steps=1] elapsed = 37.37759mn
    Epoch[3690] : PPL = 1526.31 [# Gibbs steps=1] elapsed = 37.47810mn
    Epoch[3700] : PPL = 1565.26 [# Gibbs steps=1] elapsed = 37.57525mn
    Epoch[3710] : PPL = 1165.30 [# Gibbs steps=1] elapsed = 37.67510mn
    Epoch[3720] : PPL = 1365.41 [# Gibbs steps=1] elapsed = 37.77365mn
    Epoch[3730] : PPL = 1038.02 [# Gibbs steps=1] elapsed = 37.87042mn
    Epoch[3740] : PPL = 1343.08 [# Gibbs steps=1] elapsed = 37.96635mn
    Epoch[3750] : PPL = 1288.51 [# Gibbs steps=1] elapsed = 38.06226mn
    Epoch[3760] : PPL = 1590.42 [# Gibbs steps=1] elapsed = 38.16132mn
    Epoch[3770] : PPL = 1449.54 [# Gibbs steps=1] elapsed = 38.25886mn
    Epoch[3780] : PPL = 1245.35 [# Gibbs steps=1] elapsed = 38.35892mn
    Epoch[3790] : PPL = 901.64 [# Gibbs steps=1] elapsed = 38.45381mn
    Epoch[3800] : PPL = 1428.11 [# Gibbs steps=1] elapsed = 38.54994mn
    Epoch[3810] : PPL = 1378.85 [# Gibbs steps=1] elapsed = 38.64717mn
    Epoch[3820] : PPL = 1267.81 [# Gibbs steps=1] elapsed = 38.74431mn
    Epoch[3830] : PPL = 1756.41 [# Gibbs steps=1] elapsed = 38.84394mn
    Epoch[3840] : PPL = 1326.31 [# Gibbs steps=1] elapsed = 38.94394mn
    Epoch[3850] : PPL = 1028.47 [# Gibbs steps=1] elapsed = 39.04059mn
    Epoch[3860] : PPL = 1518.84 [# Gibbs steps=1] elapsed = 39.13970mn
    Epoch[3870] : PPL = 1116.92 [# Gibbs steps=1] elapsed = 39.23777mn
    Epoch[3880] : PPL = 1358.10 [# Gibbs steps=1] elapsed = 39.33311mn
    Epoch[3890] : PPL = 1474.34 [# Gibbs steps=1] elapsed = 39.43274mn
    Epoch[3900] : PPL = 1214.22 [# Gibbs steps=1] elapsed = 39.52933mn
    Epoch[3910] : PPL = 1243.29 [# Gibbs steps=1] elapsed = 39.62918mn
    Epoch[3920] : PPL = 1974.24 [# Gibbs steps=1] elapsed = 39.72456mn
    Epoch[3930] : PPL = 1273.56 [# Gibbs steps=1] elapsed = 39.82528mn
    Epoch[3940] : PPL = 1293.62 [# Gibbs steps=1] elapsed = 39.92529mn
    Epoch[3950] : PPL = 1208.90 [# Gibbs steps=1] elapsed = 40.02381mn
    Epoch[3960] : PPL = 1450.37 [# Gibbs steps=1] elapsed = 40.12313mn
    Epoch[3970] : PPL = 1956.81 [# Gibbs steps=1] elapsed = 40.22016mn
    Epoch[3980] : PPL = 1821.42 [# Gibbs steps=1] elapsed = 40.31708mn
    Epoch[3990] : PPL = 1365.05 [# Gibbs steps=1] elapsed = 40.41390mn
    Epoch[4000] : PPL = 1531.19 [# Gibbs steps=1] elapsed = 40.68989mn
    Epoch[4010] : PPL = 1315.84 [# Gibbs steps=1] elapsed = 40.78570mn
    Epoch[4020] : PPL = 1125.29 [# Gibbs steps=1] elapsed = 40.88388mn
    Epoch[4030] : PPL = 1384.34 [# Gibbs steps=1] elapsed = 40.98190mn
    Epoch[4040] : PPL = 1589.58 [# Gibbs steps=1] elapsed = 41.07910mn
    Epoch[4050] : PPL = 1385.52 [# Gibbs steps=1] elapsed = 41.17912mn
    Epoch[4060] : PPL = 1228.74 [# Gibbs steps=1] elapsed = 41.27975mn
    Epoch[4070] : PPL = 1457.21 [# Gibbs steps=1] elapsed = 41.37930mn
    Epoch[4080] : PPL = 1452.29 [# Gibbs steps=1] elapsed = 41.47793mn
    Epoch[4090] : PPL = 1096.94 [# Gibbs steps=1] elapsed = 41.57446mn
    Epoch[4100] : PPL = 1407.86 [# Gibbs steps=1] elapsed = 41.66946mn
    Epoch[4110] : PPL = 1424.66 [# Gibbs steps=1] elapsed = 41.77002mn
    Epoch[4120] : PPL = 1313.77 [# Gibbs steps=1] elapsed = 41.86515mn
    Epoch[4130] : PPL = 981.02 [# Gibbs steps=1] elapsed = 41.96570mn
    Epoch[4140] : PPL = 1117.77 [# Gibbs steps=1] elapsed = 42.06210mn
    Epoch[4150] : PPL = 1448.89 [# Gibbs steps=1] elapsed = 42.16211mn
    Epoch[4160] : PPL = 1553.28 [# Gibbs steps=1] elapsed = 42.25985mn
    Epoch[4170] : PPL = 1502.14 [# Gibbs steps=1] elapsed = 42.35973mn
    Epoch[4180] : PPL = 801.58 [# Gibbs steps=1] elapsed = 42.45199mn
    Epoch[4190] : PPL = 1434.40 [# Gibbs steps=1] elapsed = 42.55031mn
    Epoch[4200] : PPL = 1256.18 [# Gibbs steps=1] elapsed = 42.64863mn
    Epoch[4210] : PPL = 1421.46 [# Gibbs steps=1] elapsed = 42.74738mn
    Epoch[4220] : PPL = 1441.50 [# Gibbs steps=1] elapsed = 42.84634mn
    Epoch[4230] : PPL = 1455.63 [# Gibbs steps=1] elapsed = 42.94377mn
    Epoch[4240] : PPL = 1048.45 [# Gibbs steps=1] elapsed = 43.04264mn
    Epoch[4250] : PPL = 1261.41 [# Gibbs steps=1] elapsed = 43.14300mn
    Epoch[4260] : PPL = 928.95 [# Gibbs steps=1] elapsed = 43.24255mn
    Epoch[4270] : PPL = 1400.76 [# Gibbs steps=1] elapsed = 43.34119mn
    Epoch[4280] : PPL = 1462.24 [# Gibbs steps=1] elapsed = 43.43984mn
    Epoch[4290] : PPL = 1558.37 [# Gibbs steps=1] elapsed = 43.53861mn
    Epoch[4300] : PPL = 901.27 [# Gibbs steps=1] elapsed = 43.63530mn
    Epoch[4310] : PPL = 1921.90 [# Gibbs steps=1] elapsed = 43.73342mn
    Epoch[4320] : PPL = 985.96 [# Gibbs steps=1] elapsed = 43.82993mn
    Epoch[4330] : PPL = 1441.64 [# Gibbs steps=1] elapsed = 43.92739mn
    Epoch[4340] : PPL = 1440.57 [# Gibbs steps=1] elapsed = 44.02869mn
    Epoch[4350] : PPL = 989.42 [# Gibbs steps=1] elapsed = 44.12759mn
    Epoch[4360] : PPL = 1431.27 [# Gibbs steps=1] elapsed = 44.22888mn
    Epoch[4370] : PPL = 1383.89 [# Gibbs steps=1] elapsed = 44.32891mn
    Epoch[4380] : PPL = 1367.59 [# Gibbs steps=1] elapsed = 44.42852mn
    Epoch[4390] : PPL = 1522.06 [# Gibbs steps=1] elapsed = 44.52555mn
    Epoch[4400] : PPL = 1678.50 [# Gibbs steps=1] elapsed = 44.62604mn
    Epoch[4410] : PPL = 1324.46 [# Gibbs steps=1] elapsed = 44.72344mn
    Epoch[4420] : PPL = 866.49 [# Gibbs steps=1] elapsed = 44.81681mn
    Epoch[4430] : PPL = 1520.02 [# Gibbs steps=1] elapsed = 44.91580mn
    Epoch[4440] : PPL = 1578.69 [# Gibbs steps=1] elapsed = 45.01156mn
    Epoch[4450] : PPL = 1345.85 [# Gibbs steps=1] elapsed = 45.11010mn
    Epoch[4460] : PPL = 1905.32 [# Gibbs steps=1] elapsed = 45.20959mn
    Epoch[4470] : PPL = 897.40 [# Gibbs steps=1] elapsed = 45.30581mn
    Epoch[4480] : PPL = 1431.59 [# Gibbs steps=1] elapsed = 45.40409mn
    Epoch[4490] : PPL = 1430.44 [# Gibbs steps=1] elapsed = 45.50386mn
    Epoch[4500] : PPL = 919.03 [# Gibbs steps=1] elapsed = 45.77979mn
    Epoch[4510] : PPL = 1527.36 [# Gibbs steps=1] elapsed = 45.87771mn
    Epoch[4520] : PPL = 1512.53 [# Gibbs steps=1] elapsed = 45.97587mn
    Epoch[4530] : PPL = 1410.32 [# Gibbs steps=1] elapsed = 46.07511mn
    Epoch[4540] : PPL = 1604.42 [# Gibbs steps=1] elapsed = 46.17055mn
    Epoch[4550] : PPL = 915.51 [# Gibbs steps=1] elapsed = 46.27127mn
    Epoch[4560] : PPL = 1369.23 [# Gibbs steps=1] elapsed = 46.36971mn
    Epoch[4570] : PPL = 1351.72 [# Gibbs steps=1] elapsed = 46.46902mn
    Epoch[4580] : PPL = 1520.31 [# Gibbs steps=1] elapsed = 46.56628mn
    Epoch[4590] : PPL = 1725.54 [# Gibbs steps=1] elapsed = 46.66495mn
    Epoch[4600] : PPL = 1284.20 [# Gibbs steps=1] elapsed = 46.76554mn
    Epoch[4610] : PPL = 1495.25 [# Gibbs steps=1] elapsed = 46.86300mn
    Epoch[4620] : PPL = 1412.08 [# Gibbs steps=1] elapsed = 46.96302mn
    Epoch[4630] : PPL = 1567.22 [# Gibbs steps=1] elapsed = 47.06414mn
    Epoch[4640] : PPL = 1414.83 [# Gibbs steps=1] elapsed = 47.16415mn
    Epoch[4650] : PPL = 1450.57 [# Gibbs steps=1] elapsed = 47.26504mn
    Epoch[4660] : PPL = 1530.51 [# Gibbs steps=1] elapsed = 47.36452mn
    Epoch[4670] : PPL = 1282.13 [# Gibbs steps=1] elapsed = 47.46384mn
    Epoch[4680] : PPL = 908.26 [# Gibbs steps=1] elapsed = 47.56305mn
    Epoch[4690] : PPL = 1444.53 [# Gibbs steps=1] elapsed = 47.65736mn
    Epoch[4700] : PPL = 1325.49 [# Gibbs steps=1] elapsed = 47.75389mn
    Epoch[4710] : PPL = 1375.46 [# Gibbs steps=1] elapsed = 47.85491mn
    Epoch[4720] : PPL = 1406.88 [# Gibbs steps=1] elapsed = 47.95539mn
    Epoch[4730] : PPL = 1371.58 [# Gibbs steps=1] elapsed = 48.05332mn
    Epoch[4740] : PPL = 1133.75 [# Gibbs steps=1] elapsed = 48.15315mn
    Epoch[4750] : PPL = 1424.13 [# Gibbs steps=1] elapsed = 48.25227mn
    Epoch[4760] : PPL = 1476.50 [# Gibbs steps=1] elapsed = 48.35022mn
    Epoch[4770] : PPL = 1375.73 [# Gibbs steps=1] elapsed = 48.45037mn
    Epoch[4780] : PPL = 1386.49 [# Gibbs steps=1] elapsed = 48.54729mn
    Epoch[4790] : PPL = 1251.30 [# Gibbs steps=1] elapsed = 48.64179mn
    Epoch[4800] : PPL = 1459.23 [# Gibbs steps=1] elapsed = 48.74147mn
    Epoch[4810] : PPL = 1437.38 [# Gibbs steps=1] elapsed = 48.84282mn
    Epoch[4820] : PPL = 1402.28 [# Gibbs steps=1] elapsed = 48.93947mn
    Epoch[4830] : PPL = 1387.39 [# Gibbs steps=1] elapsed = 49.04176mn
    Epoch[4840] : PPL = 1417.31 [# Gibbs steps=1] elapsed = 49.13993mn
    Epoch[4850] : PPL = 1576.11 [# Gibbs steps=1] elapsed = 49.23702mn
    Epoch[4860] : PPL = 1546.37 [# Gibbs steps=1] elapsed = 49.33266mn
    Epoch[4870] : PPL = 1539.61 [# Gibbs steps=1] elapsed = 49.43115mn
    Epoch[4880] : PPL = 954.71 [# Gibbs steps=1] elapsed = 49.53150mn
    Epoch[4890] : PPL = 1256.23 [# Gibbs steps=1] elapsed = 49.63005mn
    Epoch[4900] : PPL = 1364.36 [# Gibbs steps=1] elapsed = 49.72888mn
    Epoch[4910] : PPL = 1499.84 [# Gibbs steps=1] elapsed = 49.82322mn
    Epoch[4920] : PPL = 1109.74 [# Gibbs steps=1] elapsed = 49.91691mn
    Epoch[4930] : PPL = 1546.80 [# Gibbs steps=1] elapsed = 50.01457mn
    Epoch[4940] : PPL = 1270.72 [# Gibbs steps=1] elapsed = 50.11178mn
    Epoch[4950] : PPL = 1271.26 [# Gibbs steps=1] elapsed = 50.20959mn
    Epoch[4960] : PPL = 1403.25 [# Gibbs steps=1] elapsed = 50.31140mn
    Epoch[4970] : PPL = 1315.38 [# Gibbs steps=1] elapsed = 50.41073mn
    Epoch[4980] : PPL = 1354.52 [# Gibbs steps=1] elapsed = 50.50703mn
    Epoch[4990] : PPL = 1530.38 [# Gibbs steps=1] elapsed = 50.60617mn
    Epoch[5000] : PPL = 1611.79 [# Gibbs steps=2] elapsed = 50.88954mn
    Epoch[5010] : PPL = 1025.73 [# Gibbs steps=2] elapsed = 51.05267mn
    Epoch[5020] : PPL = 1267.21 [# Gibbs steps=2] elapsed = 51.21174mn
    Epoch[5030] : PPL = 1337.61 [# Gibbs steps=2] elapsed = 51.37578mn
    Epoch[5040] : PPL = 995.82 [# Gibbs steps=2] elapsed = 51.53694mn
    Epoch[5050] : PPL = 1394.82 [# Gibbs steps=2] elapsed = 51.69801mn
    Epoch[5060] : PPL = 1441.87 [# Gibbs steps=2] elapsed = 51.86369mn
    Epoch[5070] : PPL = 1274.01 [# Gibbs steps=2] elapsed = 52.02916mn
    Epoch[5080] : PPL = 1280.46 [# Gibbs steps=2] elapsed = 52.18860mn
    Epoch[5090] : PPL = 885.89 [# Gibbs steps=2] elapsed = 52.35379mn
    Epoch[5100] : PPL = 1146.89 [# Gibbs steps=2] elapsed = 52.51243mn
    Epoch[5110] : PPL = 1566.51 [# Gibbs steps=2] elapsed = 52.67457mn
    Epoch[5120] : PPL = 1077.47 [# Gibbs steps=2] elapsed = 52.83486mn
    Epoch[5130] : PPL = 1252.51 [# Gibbs steps=2] elapsed = 52.99505mn
    Epoch[5140] : PPL = 1927.60 [# Gibbs steps=2] elapsed = 53.16111mn
    Epoch[5150] : PPL = 1461.23 [# Gibbs steps=2] elapsed = 53.32110mn
    Epoch[5160] : PPL = 841.25 [# Gibbs steps=2] elapsed = 53.47682mn
    Epoch[5170] : PPL = 846.57 [# Gibbs steps=2] elapsed = 53.63202mn
    Epoch[5180] : PPL = 1438.20 [# Gibbs steps=2] elapsed = 53.79287mn
    Epoch[5190] : PPL = 1537.95 [# Gibbs steps=2] elapsed = 53.95778mn
    Epoch[5200] : PPL = 1290.25 [# Gibbs steps=2] elapsed = 54.11699mn
    Epoch[5210] : PPL = 1615.54 [# Gibbs steps=2] elapsed = 54.27536mn
    Epoch[5220] : PPL = 1456.15 [# Gibbs steps=2] elapsed = 54.43755mn
    Epoch[5230] : PPL = 1363.71 [# Gibbs steps=2] elapsed = 54.59847mn
    Epoch[5240] : PPL = 1268.35 [# Gibbs steps=2] elapsed = 54.76150mn
    Epoch[5250] : PPL = 1031.96 [# Gibbs steps=2] elapsed = 54.92572mn
    Epoch[5260] : PPL = 1095.29 [# Gibbs steps=2] elapsed = 55.08884mn
    Epoch[5270] : PPL = 949.97 [# Gibbs steps=2] elapsed = 55.24829mn
    Epoch[5280] : PPL = 1589.76 [# Gibbs steps=2] elapsed = 55.41214mn
    Epoch[5290] : PPL = 1194.38 [# Gibbs steps=2] elapsed = 55.57496mn
    Epoch[5300] : PPL = 1367.26 [# Gibbs steps=2] elapsed = 55.73525mn
    Epoch[5310] : PPL = 1215.24 [# Gibbs steps=2] elapsed = 55.89658mn
    Epoch[5320] : PPL = 1927.29 [# Gibbs steps=2] elapsed = 56.05895mn
    Epoch[5330] : PPL = 1359.06 [# Gibbs steps=2] elapsed = 56.22088mn
    Epoch[5340] : PPL = 1219.27 [# Gibbs steps=2] elapsed = 56.38370mn
    Epoch[5350] : PPL = 1268.04 [# Gibbs steps=2] elapsed = 56.54520mn
    Epoch[5360] : PPL = 1162.71 [# Gibbs steps=2] elapsed = 56.70425mn
    Epoch[5370] : PPL = 1562.18 [# Gibbs steps=2] elapsed = 56.86750mn
    Epoch[5380] : PPL = 1273.87 [# Gibbs steps=2] elapsed = 57.03179mn
    Epoch[5390] : PPL = 1594.87 [# Gibbs steps=2] elapsed = 57.19465mn
    Epoch[5400] : PPL = 1113.68 [# Gibbs steps=2] elapsed = 57.35845mn
    Epoch[5410] : PPL = 1457.59 [# Gibbs steps=2] elapsed = 57.51955mn
    Epoch[5420] : PPL = 904.16 [# Gibbs steps=2] elapsed = 57.68393mn
    Epoch[5430] : PPL = 1299.32 [# Gibbs steps=2] elapsed = 57.84161mn
    Epoch[5440] : PPL = 1294.41 [# Gibbs steps=2] elapsed = 57.99821mn
    Epoch[5450] : PPL = 1544.25 [# Gibbs steps=2] elapsed = 58.15986mn
    Epoch[5460] : PPL = 839.64 [# Gibbs steps=2] elapsed = 58.32136mn
    Epoch[5470] : PPL = 1264.55 [# Gibbs steps=2] elapsed = 58.47925mn
    Epoch[5480] : PPL = 1006.75 [# Gibbs steps=2] elapsed = 58.64082mn
    Epoch[5490] : PPL = 1461.89 [# Gibbs steps=2] elapsed = 58.80645mn
    Epoch[5500] : PPL = 916.72 [# Gibbs steps=2] elapsed = 59.14781mn
    Epoch[5510] : PPL = 1344.79 [# Gibbs steps=2] elapsed = 59.31378mn
    Epoch[5520] : PPL = 1406.02 [# Gibbs steps=2] elapsed = 59.47738mn
    Epoch[5530] : PPL = 1232.83 [# Gibbs steps=2] elapsed = 59.64011mn
    Epoch[5540] : PPL = 1628.03 [# Gibbs steps=2] elapsed = 59.80408mn
    Epoch[5550] : PPL = 919.52 [# Gibbs steps=2] elapsed = 59.96737mn
    Epoch[5560] : PPL = 1609.94 [# Gibbs steps=2] elapsed = 60.12764mn
    Epoch[5570] : PPL = 1070.54 [# Gibbs steps=2] elapsed = 60.29019mn
    Epoch[5580] : PPL = 970.20 [# Gibbs steps=2] elapsed = 60.45267mn
    Epoch[5590] : PPL = 1537.70 [# Gibbs steps=2] elapsed = 60.61553mn
    Epoch[5600] : PPL = 1429.81 [# Gibbs steps=2] elapsed = 60.77525mn
    Epoch[5610] : PPL = 1490.42 [# Gibbs steps=2] elapsed = 60.93512mn
    Epoch[5620] : PPL = 1498.72 [# Gibbs steps=2] elapsed = 61.09690mn
    Epoch[5630] : PPL = 1301.07 [# Gibbs steps=2] elapsed = 61.25983mn
    Epoch[5640] : PPL = 1491.97 [# Gibbs steps=2] elapsed = 61.43022mn
    Epoch[5650] : PPL = 1280.62 [# Gibbs steps=2] elapsed = 61.59939mn
    Epoch[5660] : PPL = 1599.95 [# Gibbs steps=2] elapsed = 61.76965mn
    Epoch[5670] : PPL = 1217.64 [# Gibbs steps=2] elapsed = 61.93946mn
    Epoch[5680] : PPL = 906.77 [# Gibbs steps=2] elapsed = 62.10998mn
    Epoch[5690] : PPL = 1444.82 [# Gibbs steps=2] elapsed = 62.28083mn
    Epoch[5700] : PPL = 1411.57 [# Gibbs steps=2] elapsed = 62.45139mn
    Epoch[5710] : PPL = 1595.56 [# Gibbs steps=2] elapsed = 62.61969mn
    Epoch[5720] : PPL = 1430.46 [# Gibbs steps=2] elapsed = 62.78530mn
    Epoch[5730] : PPL = 1266.74 [# Gibbs steps=2] elapsed = 62.94750mn
    Epoch[5740] : PPL = 1494.05 [# Gibbs steps=2] elapsed = 63.11951mn
    Epoch[5750] : PPL = 945.51 [# Gibbs steps=2] elapsed = 63.29067mn
    Epoch[5760] : PPL = 1463.40 [# Gibbs steps=2] elapsed = 63.45912mn
    Epoch[5770] : PPL = 1647.91 [# Gibbs steps=2] elapsed = 63.62508mn
    Epoch[5780] : PPL = 1303.69 [# Gibbs steps=2] elapsed = 63.79320mn
    Epoch[5790] : PPL = 1114.21 [# Gibbs steps=2] elapsed = 63.96391mn
    Epoch[5800] : PPL = 1421.13 [# Gibbs steps=2] elapsed = 64.13172mn
    Epoch[5810] : PPL = 1478.77 [# Gibbs steps=2] elapsed = 64.30353mn
    Epoch[5820] : PPL = 1316.57 [# Gibbs steps=2] elapsed = 64.46950mn
    Epoch[5830] : PPL = 1506.05 [# Gibbs steps=2] elapsed = 64.64078mn
    Epoch[5840] : PPL = 1407.17 [# Gibbs steps=2] elapsed = 64.81303mn
    Epoch[5850] : PPL = 1484.30 [# Gibbs steps=2] elapsed = 64.98259mn
    Epoch[5860] : PPL = 1351.64 [# Gibbs steps=2] elapsed = 65.15301mn
    Epoch[5870] : PPL = 1084.85 [# Gibbs steps=2] elapsed = 65.32316mn
    Epoch[5880] : PPL = 1468.65 [# Gibbs steps=2] elapsed = 65.49462mn
    Epoch[5890] : PPL = 1405.90 [# Gibbs steps=2] elapsed = 65.66646mn
    Epoch[5900] : PPL = 1450.19 [# Gibbs steps=2] elapsed = 65.83297mn
    Epoch[5910] : PPL = 1386.03 [# Gibbs steps=2] elapsed = 65.99935mn
    Epoch[5920] : PPL = 1648.84 [# Gibbs steps=2] elapsed = 66.17022mn
    Epoch[5930] : PPL = 1284.04 [# Gibbs steps=2] elapsed = 66.34086mn
    Epoch[5940] : PPL = 1521.76 [# Gibbs steps=2] elapsed = 66.51211mn
    Epoch[5950] : PPL = 1318.38 [# Gibbs steps=2] elapsed = 66.68287mn
    Epoch[5960] : PPL = 853.09 [# Gibbs steps=2] elapsed = 66.85218mn
    Epoch[5970] : PPL = 1539.45 [# Gibbs steps=2] elapsed = 67.02425mn
    Epoch[5980] : PPL = 1008.67 [# Gibbs steps=2] elapsed = 67.19229mn
    Epoch[5990] : PPL = 911.76 [# Gibbs steps=2] elapsed = 67.35870mn
    Epoch[6000] : PPL = 1348.58 [# Gibbs steps=2] elapsed = 67.71238mn
    Epoch[6010] : PPL = 1648.68 [# Gibbs steps=2] elapsed = 67.88071mn
    Epoch[6020] : PPL = 1347.71 [# Gibbs steps=2] elapsed = 68.04583mn
    Epoch[6030] : PPL = 1295.24 [# Gibbs steps=2] elapsed = 68.21826mn
    Epoch[6040] : PPL = 1295.77 [# Gibbs steps=2] elapsed = 68.38743mn
    Epoch[6050] : PPL = 913.03 [# Gibbs steps=2] elapsed = 68.55893mn
    Epoch[6060] : PPL = 1553.13 [# Gibbs steps=2] elapsed = 68.72660mn
    Epoch[6070] : PPL = 917.25 [# Gibbs steps=2] elapsed = 68.89280mn
    Epoch[6080] : PPL = 1198.86 [# Gibbs steps=2] elapsed = 69.06250mn
    Epoch[6090] : PPL = 1469.86 [# Gibbs steps=2] elapsed = 69.23090mn
    Epoch[6100] : PPL = 1318.23 [# Gibbs steps=2] elapsed = 69.39837mn
    Epoch[6110] : PPL = 1288.10 [# Gibbs steps=2] elapsed = 69.57002mn
    Epoch[6120] : PPL = 1376.66 [# Gibbs steps=2] elapsed = 69.73803mn
    Epoch[6130] : PPL = 1536.89 [# Gibbs steps=2] elapsed = 69.90834mn
    Epoch[6140] : PPL = 1461.24 [# Gibbs steps=2] elapsed = 70.07767mn
    Epoch[6150] : PPL = 1411.66 [# Gibbs steps=2] elapsed = 70.24494mn
    Epoch[6160] : PPL = 1148.00 [# Gibbs steps=2] elapsed = 70.41693mn
    Epoch[6170] : PPL = 1022.50 [# Gibbs steps=2] elapsed = 70.58670mn
    Epoch[6180] : PPL = 1515.21 [# Gibbs steps=2] elapsed = 70.75552mn
    Epoch[6190] : PPL = 1607.20 [# Gibbs steps=2] elapsed = 70.92232mn
    Epoch[6200] : PPL = 1354.66 [# Gibbs steps=2] elapsed = 71.09130mn
    Epoch[6210] : PPL = 1611.29 [# Gibbs steps=2] elapsed = 71.25769mn
    Epoch[6220] : PPL = 1295.34 [# Gibbs steps=2] elapsed = 71.42452mn
    Epoch[6230] : PPL = 2014.03 [# Gibbs steps=2] elapsed = 71.59483mn
    Epoch[6240] : PPL = 1318.93 [# Gibbs steps=2] elapsed = 71.76513mn
    Epoch[6250] : PPL = 1428.25 [# Gibbs steps=2] elapsed = 71.93666mn
    Epoch[6260] : PPL = 1043.18 [# Gibbs steps=2] elapsed = 72.10608mn
    Epoch[6270] : PPL = 1470.67 [# Gibbs steps=2] elapsed = 72.27603mn
    Epoch[6280] : PPL = 1206.79 [# Gibbs steps=2] elapsed = 72.44575mn
    Epoch[6290] : PPL = 1463.19 [# Gibbs steps=2] elapsed = 72.61575mn
    Epoch[6300] : PPL = 1042.54 [# Gibbs steps=2] elapsed = 72.78739mn
    Epoch[6310] : PPL = 1519.11 [# Gibbs steps=2] elapsed = 72.95483mn
    Epoch[6320] : PPL = 1509.94 [# Gibbs steps=2] elapsed = 73.12425mn
    Epoch[6330] : PPL = 1326.22 [# Gibbs steps=2] elapsed = 73.29267mn
    Epoch[6340] : PPL = 1545.72 [# Gibbs steps=2] elapsed = 73.46446mn
    Epoch[6350] : PPL = 1475.05 [# Gibbs steps=2] elapsed = 73.63301mn
    Epoch[6360] : PPL = 1288.57 [# Gibbs steps=2] elapsed = 73.80114mn
    Epoch[6370] : PPL = 888.35 [# Gibbs steps=2] elapsed = 73.96981mn
    Epoch[6380] : PPL = 1354.48 [# Gibbs steps=2] elapsed = 74.13930mn
    Epoch[6390] : PPL = 1537.18 [# Gibbs steps=2] elapsed = 74.30918mn
    Epoch[6400] : PPL = 1590.05 [# Gibbs steps=2] elapsed = 74.48062mn
    Epoch[6410] : PPL = 1291.67 [# Gibbs steps=2] elapsed = 74.64748mn
    Epoch[6420] : PPL = 1314.26 [# Gibbs steps=2] elapsed = 74.81109mn
    Epoch[6430] : PPL = 1457.79 [# Gibbs steps=2] elapsed = 74.97897mn
    Epoch[6440] : PPL = 1456.02 [# Gibbs steps=2] elapsed = 75.14982mn
    Epoch[6450] : PPL = 1199.24 [# Gibbs steps=2] elapsed = 75.32029mn
    Epoch[6460] : PPL = 1927.50 [# Gibbs steps=2] elapsed = 75.48949mn
    Epoch[6470] : PPL = 954.77 [# Gibbs steps=2] elapsed = 75.65834mn
    Epoch[6480] : PPL = 1319.02 [# Gibbs steps=2] elapsed = 75.83098mn
    Epoch[6490] : PPL = 1302.81 [# Gibbs steps=2] elapsed = 76.00062mn
    Epoch[6500] : PPL = 1269.01 [# Gibbs steps=2] elapsed = 76.34166mn
    Epoch[6510] : PPL = 1664.68 [# Gibbs steps=2] elapsed = 76.50292mn
    Epoch[6520] : PPL = 1558.06 [# Gibbs steps=2] elapsed = 76.66467mn
    Epoch[6530] : PPL = 1105.34 [# Gibbs steps=2] elapsed = 76.82625mn
    Epoch[6540] : PPL = 888.19 [# Gibbs steps=2] elapsed = 76.98637mn
    Epoch[6550] : PPL = 1347.58 [# Gibbs steps=2] elapsed = 77.14738mn
    Epoch[6560] : PPL = 1313.19 [# Gibbs steps=2] elapsed = 77.30994mn
    Epoch[6570] : PPL = 1452.41 [# Gibbs steps=2] elapsed = 77.47299mn
    Epoch[6580] : PPL = 1400.93 [# Gibbs steps=2] elapsed = 77.63433mn
    Epoch[6590] : PPL = 1331.98 [# Gibbs steps=2] elapsed = 77.79147mn
    Epoch[6600] : PPL = 1310.55 [# Gibbs steps=2] elapsed = 77.95057mn
    Epoch[6610] : PPL = 1427.60 [# Gibbs steps=2] elapsed = 78.11335mn
    Epoch[6620] : PPL = 1613.70 [# Gibbs steps=2] elapsed = 78.27780mn
    Epoch[6630] : PPL = 1358.63 [# Gibbs steps=2] elapsed = 78.43929mn
    Epoch[6640] : PPL = 1486.30 [# Gibbs steps=2] elapsed = 78.60060mn
    Epoch[6650] : PPL = 1699.89 [# Gibbs steps=2] elapsed = 78.76080mn
    Epoch[6660] : PPL = 1581.52 [# Gibbs steps=2] elapsed = 78.92258mn
    Epoch[6670] : PPL = 1511.11 [# Gibbs steps=2] elapsed = 79.08538mn
    Epoch[6680] : PPL = 1585.47 [# Gibbs steps=2] elapsed = 79.24779mn
    Epoch[6690] : PPL = 1306.57 [# Gibbs steps=2] elapsed = 79.40972mn
    Epoch[6700] : PPL = 1501.82 [# Gibbs steps=2] elapsed = 79.57076mn
    Epoch[6710] : PPL = 1508.76 [# Gibbs steps=2] elapsed = 79.73189mn
    Epoch[6720] : PPL = 1294.65 [# Gibbs steps=2] elapsed = 79.89179mn
    Epoch[6730] : PPL = 879.45 [# Gibbs steps=2] elapsed = 80.05389mn
    Epoch[6740] : PPL = 1531.92 [# Gibbs steps=2] elapsed = 80.21675mn
    Epoch[6750] : PPL = 880.90 [# Gibbs steps=2] elapsed = 80.37348mn
    Epoch[6760] : PPL = 1495.17 [# Gibbs steps=2] elapsed = 80.53669mn
    Epoch[6770] : PPL = 1448.65 [# Gibbs steps=2] elapsed = 80.69983mn
    Epoch[6780] : PPL = 1548.95 [# Gibbs steps=2] elapsed = 80.86214mn
    Epoch[6790] : PPL = 1217.10 [# Gibbs steps=2] elapsed = 81.02256mn
    Epoch[6800] : PPL = 1152.13 [# Gibbs steps=2] elapsed = 81.18365mn
    Epoch[6810] : PPL = 1312.13 [# Gibbs steps=2] elapsed = 81.34291mn
    Epoch[6820] : PPL = 1220.72 [# Gibbs steps=2] elapsed = 81.50519mn
    Epoch[6830] : PPL = 1528.57 [# Gibbs steps=2] elapsed = 81.67074mn
    Epoch[6840] : PPL = 1704.12 [# Gibbs steps=2] elapsed = 81.83067mn
    Epoch[6850] : PPL = 873.24 [# Gibbs steps=2] elapsed = 81.98851mn
    Epoch[6860] : PPL = 1062.95 [# Gibbs steps=2] elapsed = 82.14919mn
    Epoch[6870] : PPL = 1357.68 [# Gibbs steps=2] elapsed = 82.31396mn
    Epoch[6880] : PPL = 1069.52 [# Gibbs steps=2] elapsed = 82.47600mn
    Epoch[6890] : PPL = 1559.21 [# Gibbs steps=2] elapsed = 82.64112mn
    Epoch[6900] : PPL = 877.18 [# Gibbs steps=2] elapsed = 82.80163mn
    Epoch[6910] : PPL = 1516.48 [# Gibbs steps=2] elapsed = 82.96263mn
    Epoch[6920] : PPL = 1504.64 [# Gibbs steps=2] elapsed = 83.12507mn
    Epoch[6930] : PPL = 1541.20 [# Gibbs steps=2] elapsed = 83.28579mn
    Epoch[6940] : PPL = 1413.84 [# Gibbs steps=2] elapsed = 83.44942mn
    Epoch[6950] : PPL = 1493.64 [# Gibbs steps=2] elapsed = 83.60994mn
    Epoch[6960] : PPL = 1308.40 [# Gibbs steps=2] elapsed = 83.77001mn
    Epoch[6970] : PPL = 1267.36 [# Gibbs steps=2] elapsed = 83.92672mn
    Epoch[6980] : PPL = 1507.60 [# Gibbs steps=2] elapsed = 84.09102mn
    Epoch[6990] : PPL = 1525.07 [# Gibbs steps=2] elapsed = 84.25743mn
    Epoch[7000] : PPL = 1485.68 [# Gibbs steps=2] elapsed = 84.59513mn
    Epoch[7010] : PPL = 1265.69 [# Gibbs steps=2] elapsed = 84.75993mn
    Epoch[7020] : PPL = 949.33 [# Gibbs steps=2] elapsed = 84.92105mn
    Epoch[7030] : PPL = 1459.58 [# Gibbs steps=2] elapsed = 85.08121mn
    Epoch[7040] : PPL = 1335.15 [# Gibbs steps=2] elapsed = 85.24080mn
    Epoch[7050] : PPL = 1465.26 [# Gibbs steps=2] elapsed = 85.40387mn
    Epoch[7060] : PPL = 1532.93 [# Gibbs steps=2] elapsed = 85.56601mn
    Epoch[7070] : PPL = 1569.59 [# Gibbs steps=2] elapsed = 85.73027mn
    Epoch[7080] : PPL = 1306.02 [# Gibbs steps=2] elapsed = 85.89125mn
    Epoch[7090] : PPL = 1480.79 [# Gibbs steps=2] elapsed = 86.05318mn
    Epoch[7100] : PPL = 1530.62 [# Gibbs steps=2] elapsed = 86.21623mn
    Epoch[7110] : PPL = 1364.84 [# Gibbs steps=2] elapsed = 86.38063mn
    Epoch[7120] : PPL = 923.22 [# Gibbs steps=2] elapsed = 86.54195mn
    Epoch[7130] : PPL = 1119.26 [# Gibbs steps=2] elapsed = 86.70525mn
    Epoch[7140] : PPL = 1312.58 [# Gibbs steps=2] elapsed = 86.86878mn
    Epoch[7150] : PPL = 1446.02 [# Gibbs steps=2] elapsed = 87.03156mn
    Epoch[7160] : PPL = 1345.45 [# Gibbs steps=2] elapsed = 87.19406mn
    Epoch[7170] : PPL = 1956.16 [# Gibbs steps=2] elapsed = 87.35515mn
    Epoch[7180] : PPL = 1537.62 [# Gibbs steps=2] elapsed = 87.51916mn
    Epoch[7190] : PPL = 1076.80 [# Gibbs steps=2] elapsed = 87.68034mn
    Epoch[7200] : PPL = 1292.35 [# Gibbs steps=2] elapsed = 87.84359mn
    Epoch[7210] : PPL = 1341.05 [# Gibbs steps=2] elapsed = 88.00603mn
    Epoch[7220] : PPL = 1554.34 [# Gibbs steps=2] elapsed = 88.16717mn
    Epoch[7230] : PPL = 1564.75 [# Gibbs steps=2] elapsed = 88.32816mn
    Epoch[7240] : PPL = 1492.22 [# Gibbs steps=2] elapsed = 88.48688mn
    Epoch[7250] : PPL = 1460.06 [# Gibbs steps=2] elapsed = 88.65231mn
    Epoch[7260] : PPL = 1413.47 [# Gibbs steps=2] elapsed = 88.81513mn
    Epoch[7270] : PPL = 1914.97 [# Gibbs steps=2] elapsed = 88.97597mn
    Epoch[7280] : PPL = 1105.26 [# Gibbs steps=2] elapsed = 89.13318mn
    Epoch[7290] : PPL = 1185.52 [# Gibbs steps=2] elapsed = 89.29738mn
    Epoch[7300] : PPL = 1543.08 [# Gibbs steps=2] elapsed = 89.46204mn
    Epoch[7310] : PPL = 1555.14 [# Gibbs steps=2] elapsed = 89.62424mn
    Epoch[7320] : PPL = 1367.75 [# Gibbs steps=2] elapsed = 89.78722mn
    Epoch[7330] : PPL = 1139.94 [# Gibbs steps=2] elapsed = 89.94775mn
    Epoch[7340] : PPL = 1390.75 [# Gibbs steps=2] elapsed = 90.10737mn
    Epoch[7350] : PPL = 892.58 [# Gibbs steps=2] elapsed = 90.26598mn
    Epoch[7360] : PPL = 925.40 [# Gibbs steps=2] elapsed = 90.42660mn
    Epoch[7370] : PPL = 1323.86 [# Gibbs steps=2] elapsed = 90.59254mn
    Epoch[7380] : PPL = 1593.00 [# Gibbs steps=2] elapsed = 90.75806mn
    Epoch[7390] : PPL = 1498.79 [# Gibbs steps=2] elapsed = 90.91794mn
    Epoch[7400] : PPL = 922.20 [# Gibbs steps=2] elapsed = 91.07593mn
    Epoch[7410] : PPL = 1374.80 [# Gibbs steps=2] elapsed = 91.23981mn
    Epoch[7420] : PPL = 1659.83 [# Gibbs steps=2] elapsed = 91.40326mn
    Epoch[7430] : PPL = 1641.27 [# Gibbs steps=2] elapsed = 91.56277mn
    Epoch[7440] : PPL = 1661.05 [# Gibbs steps=2] elapsed = 91.72487mn
    Epoch[7450] : PPL = 1099.22 [# Gibbs steps=2] elapsed = 91.88525mn
    Epoch[7460] : PPL = 1228.15 [# Gibbs steps=2] elapsed = 92.04815mn
    Epoch[7470] : PPL = 1206.07 [# Gibbs steps=2] elapsed = 92.20839mn
    Epoch[7480] : PPL = 1954.87 [# Gibbs steps=2] elapsed = 92.37057mn
    Epoch[7490] : PPL = 1167.51 [# Gibbs steps=2] elapsed = 92.53327mn
    Epoch[7500] : PPL = 1298.90 [# Gibbs steps=2] elapsed = 92.87149mn
    Epoch[7510] : PPL = 1373.83 [# Gibbs steps=2] elapsed = 93.03545mn
    Epoch[7520] : PPL = 1267.99 [# Gibbs steps=2] elapsed = 93.19945mn
    Epoch[7530] : PPL = 1417.08 [# Gibbs steps=2] elapsed = 93.36329mn
    Epoch[7540] : PPL = 1111.75 [# Gibbs steps=2] elapsed = 93.52430mn
    Epoch[7550] : PPL = 1311.08 [# Gibbs steps=2] elapsed = 93.68714mn
    Epoch[7560] : PPL = 1162.58 [# Gibbs steps=2] elapsed = 93.85053mn
    Epoch[7570] : PPL = 1263.59 [# Gibbs steps=2] elapsed = 94.01096mn
    Epoch[7580] : PPL = 1452.02 [# Gibbs steps=2] elapsed = 94.17407mn
    Epoch[7590] : PPL = 1374.35 [# Gibbs steps=2] elapsed = 94.33773mn
    Epoch[7600] : PPL = 1527.01 [# Gibbs steps=2] elapsed = 94.50091mn
    Epoch[7610] : PPL = 1084.74 [# Gibbs steps=2] elapsed = 94.66129mn
    Epoch[7620] : PPL = 1585.67 [# Gibbs steps=2] elapsed = 94.82473mn
    Epoch[7630] : PPL = 1120.12 [# Gibbs steps=2] elapsed = 94.98177mn
    Epoch[7640] : PPL = 1261.67 [# Gibbs steps=2] elapsed = 95.14399mn
    Epoch[7650] : PPL = 1564.79 [# Gibbs steps=2] elapsed = 95.30461mn
    Epoch[7660] : PPL = 1337.66 [# Gibbs steps=2] elapsed = 95.46409mn
    Epoch[7670] : PPL = 1565.57 [# Gibbs steps=2] elapsed = 95.62357mn
    Epoch[7680] : PPL = 1548.16 [# Gibbs steps=2] elapsed = 95.78600mn
    Epoch[7690] : PPL = 1251.07 [# Gibbs steps=2] elapsed = 95.94719mn
    Epoch[7700] : PPL = 1149.40 [# Gibbs steps=2] elapsed = 96.10910mn
    Epoch[7710] : PPL = 1476.02 [# Gibbs steps=2] elapsed = 96.27256mn
    Epoch[7720] : PPL = 1388.14 [# Gibbs steps=2] elapsed = 96.43324mn
    Epoch[7730] : PPL = 1507.95 [# Gibbs steps=2] elapsed = 96.59475mn
    Epoch[7740] : PPL = 856.11 [# Gibbs steps=2] elapsed = 96.75417mn
    Epoch[7750] : PPL = 1317.87 [# Gibbs steps=2] elapsed = 96.91861mn
    Epoch[7760] : PPL = 1297.02 [# Gibbs steps=2] elapsed = 97.08017mn
    Epoch[7770] : PPL = 1613.57 [# Gibbs steps=2] elapsed = 97.24240mn
    Epoch[7780] : PPL = 1501.56 [# Gibbs steps=2] elapsed = 97.40752mn
    Epoch[7790] : PPL = 1081.57 [# Gibbs steps=2] elapsed = 97.56922mn
    Epoch[7800] : PPL = 1597.31 [# Gibbs steps=2] elapsed = 97.73378mn
    Epoch[7810] : PPL = 1738.73 [# Gibbs steps=2] elapsed = 97.89460mn
    Epoch[7820] : PPL = 841.06 [# Gibbs steps=2] elapsed = 98.05477mn
    Epoch[7830] : PPL = 1115.07 [# Gibbs steps=2] elapsed = 98.21755mn
    Epoch[7840] : PPL = 1468.82 [# Gibbs steps=2] elapsed = 98.38041mn
    Epoch[7850] : PPL = 1285.54 [# Gibbs steps=2] elapsed = 98.54299mn
    Epoch[7860] : PPL = 1306.22 [# Gibbs steps=2] elapsed = 98.70194mn
    Epoch[7870] : PPL = 1322.40 [# Gibbs steps=2] elapsed = 98.86307mn
    Epoch[7880] : PPL = 1305.67 [# Gibbs steps=2] elapsed = 99.02275mn
    Epoch[7890] : PPL = 1270.33 [# Gibbs steps=2] elapsed = 99.18441mn
    Epoch[7900] : PPL = 1139.77 [# Gibbs steps=2] elapsed = 99.34206mn
    Epoch[7910] : PPL = 1427.11 [# Gibbs steps=2] elapsed = 99.50084mn
    Epoch[7920] : PPL = 1490.50 [# Gibbs steps=2] elapsed = 99.66350mn
    Epoch[7930] : PPL = 1573.57 [# Gibbs steps=2] elapsed = 99.82270mn
    Epoch[7940] : PPL = 1434.97 [# Gibbs steps=2] elapsed = 99.98030mn
    Epoch[7950] : PPL = 1601.49 [# Gibbs steps=2] elapsed = 100.14157mn
    Epoch[7960] : PPL = 1537.26 [# Gibbs steps=2] elapsed = 100.30407mn
    Epoch[7970] : PPL = 1429.81 [# Gibbs steps=2] elapsed = 100.46682mn
    Epoch[7980] : PPL = 1043.02 [# Gibbs steps=2] elapsed = 100.62907mn
    Epoch[7990] : PPL = 1440.47 [# Gibbs steps=2] elapsed = 100.78994mn
    Epoch[8000] : PPL = 952.07 [# Gibbs steps=2] elapsed = 101.12490mn
    Epoch[8010] : PPL = 1423.73 [# Gibbs steps=2] elapsed = 101.28840mn
    Epoch[8020] : PPL = 1413.98 [# Gibbs steps=2] elapsed = 101.45192mn
    Epoch[8030] : PPL = 1497.60 [# Gibbs steps=2] elapsed = 101.61273mn
    Epoch[8040] : PPL = 1356.01 [# Gibbs steps=2] elapsed = 101.77707mn
    Epoch[8050] : PPL = 1471.96 [# Gibbs steps=2] elapsed = 101.93741mn
    Epoch[8060] : PPL = 1569.80 [# Gibbs steps=2] elapsed = 102.09744mn
    Epoch[8070] : PPL = 1371.11 [# Gibbs steps=2] elapsed = 102.25993mn
    Epoch[8080] : PPL = 1296.09 [# Gibbs steps=2] elapsed = 102.42092mn
    Epoch[8090] : PPL = 1231.62 [# Gibbs steps=2] elapsed = 102.58492mn
    Epoch[8100] : PPL = 1479.73 [# Gibbs steps=2] elapsed = 102.74563mn
    Epoch[8110] : PPL = 1528.07 [# Gibbs steps=2] elapsed = 102.90506mn
    Epoch[8120] : PPL = 1624.23 [# Gibbs steps=2] elapsed = 103.06619mn
    Epoch[8130] : PPL = 1246.02 [# Gibbs steps=2] elapsed = 103.22579mn
    Epoch[8140] : PPL = 1640.25 [# Gibbs steps=2] elapsed = 103.38922mn
    Epoch[8150] : PPL = 1376.27 [# Gibbs steps=2] elapsed = 103.54801mn
    Epoch[8160] : PPL = 1481.52 [# Gibbs steps=2] elapsed = 103.71132mn
    Epoch[8170] : PPL = 1582.21 [# Gibbs steps=2] elapsed = 103.86978mn
    Epoch[8180] : PPL = 1497.54 [# Gibbs steps=2] elapsed = 104.02874mn
    Epoch[8190] : PPL = 922.38 [# Gibbs steps=2] elapsed = 104.19067mn
    Epoch[8200] : PPL = 1306.39 [# Gibbs steps=2] elapsed = 104.35084mn
    Epoch[8210] : PPL = 1464.28 [# Gibbs steps=2] elapsed = 104.51385mn
    Epoch[8220] : PPL = 1027.84 [# Gibbs steps=2] elapsed = 104.67378mn
    Epoch[8230] : PPL = 1466.63 [# Gibbs steps=2] elapsed = 104.83538mn
    Epoch[8240] : PPL = 1080.08 [# Gibbs steps=2] elapsed = 104.99306mn
    Epoch[8250] : PPL = 1232.79 [# Gibbs steps=2] elapsed = 105.15471mn
    Epoch[8260] : PPL = 1575.95 [# Gibbs steps=2] elapsed = 105.31757mn
    Epoch[8270] : PPL = 1392.17 [# Gibbs steps=2] elapsed = 105.47902mn
    Epoch[8280] : PPL = 1161.24 [# Gibbs steps=2] elapsed = 105.64156mn
    Epoch[8290] : PPL = 1467.75 [# Gibbs steps=2] elapsed = 105.80356mn
    Epoch[8300] : PPL = 1636.07 [# Gibbs steps=2] elapsed = 105.96862mn
    Epoch[8310] : PPL = 1330.03 [# Gibbs steps=2] elapsed = 106.12925mn
    Epoch[8320] : PPL = 1226.58 [# Gibbs steps=2] elapsed = 106.29088mn
    Epoch[8330] : PPL = 1542.86 [# Gibbs steps=2] elapsed = 106.45474mn
    Epoch[8340] : PPL = 1430.42 [# Gibbs steps=2] elapsed = 106.61890mn
    Epoch[8350] : PPL = 1028.69 [# Gibbs steps=2] elapsed = 106.77914mn
    Epoch[8360] : PPL = 1550.85 [# Gibbs steps=2] elapsed = 106.94250mn
    Epoch[8370] : PPL = 1374.06 [# Gibbs steps=2] elapsed = 107.10197mn
    Epoch[8380] : PPL = 1218.55 [# Gibbs steps=2] elapsed = 107.26325mn
    Epoch[8390] : PPL = 1313.60 [# Gibbs steps=2] elapsed = 107.42506mn
    Epoch[8400] : PPL = 1193.15 [# Gibbs steps=2] elapsed = 107.58632mn
    Epoch[8410] : PPL = 1213.50 [# Gibbs steps=2] elapsed = 107.74383mn
    Epoch[8420] : PPL = 1373.12 [# Gibbs steps=2] elapsed = 107.90343mn
    Epoch[8430] : PPL = 1490.04 [# Gibbs steps=2] elapsed = 108.06780mn
    Epoch[8440] : PPL = 1462.22 [# Gibbs steps=2] elapsed = 108.23243mn
    Epoch[8450] : PPL = 1439.37 [# Gibbs steps=2] elapsed = 108.39284mn
    Epoch[8460] : PPL = 958.29 [# Gibbs steps=2] elapsed = 108.55276mn
    Epoch[8470] : PPL = 1271.99 [# Gibbs steps=2] elapsed = 108.71363mn
    Epoch[8480] : PPL = 1518.36 [# Gibbs steps=2] elapsed = 108.87302mn
    Epoch[8490] : PPL = 1488.52 [# Gibbs steps=2] elapsed = 109.03619mn
    Epoch[8500] : PPL = 1234.25 [# Gibbs steps=2] elapsed = 109.37234mn
    Epoch[8510] : PPL = 1307.48 [# Gibbs steps=2] elapsed = 109.53470mn
    Epoch[8520] : PPL = 1452.78 [# Gibbs steps=2] elapsed = 109.69627mn
    Epoch[8530] : PPL = 1280.87 [# Gibbs steps=2] elapsed = 109.86177mn
    Epoch[8540] : PPL = 887.42 [# Gibbs steps=2] elapsed = 110.02186mn
    Epoch[8550] : PPL = 1488.49 [# Gibbs steps=2] elapsed = 110.18341mn
    Epoch[8560] : PPL = 1434.83 [# Gibbs steps=2] elapsed = 110.34562mn
    Epoch[8570] : PPL = 1293.41 [# Gibbs steps=2] elapsed = 110.50521mn
    Epoch[8580] : PPL = 1355.33 [# Gibbs steps=2] elapsed = 110.66748mn
    Epoch[8590] : PPL = 1199.72 [# Gibbs steps=2] elapsed = 110.83333mn
    Epoch[8600] : PPL = 1523.43 [# Gibbs steps=2] elapsed = 110.99517mn
    Epoch[8610] : PPL = 1301.36 [# Gibbs steps=2] elapsed = 111.15497mn
    Epoch[8620] : PPL = 1158.82 [# Gibbs steps=2] elapsed = 111.31588mn
    Epoch[8630] : PPL = 1269.81 [# Gibbs steps=2] elapsed = 111.47721mn
    Epoch[8640] : PPL = 865.16 [# Gibbs steps=2] elapsed = 111.63796mn
    Epoch[8650] : PPL = 1487.53 [# Gibbs steps=2] elapsed = 111.80032mn
    Epoch[8660] : PPL = 1144.48 [# Gibbs steps=2] elapsed = 111.95915mn
    Epoch[8670] : PPL = 1932.77 [# Gibbs steps=2] elapsed = 112.12124mn
    Epoch[8680] : PPL = 1541.79 [# Gibbs steps=2] elapsed = 112.28517mn
    Epoch[8690] : PPL = 1568.81 [# Gibbs steps=2] elapsed = 112.44711mn
    Epoch[8700] : PPL = 1364.69 [# Gibbs steps=2] elapsed = 112.60818mn
    Epoch[8710] : PPL = 1148.54 [# Gibbs steps=2] elapsed = 112.77040mn
    Epoch[8720] : PPL = 935.07 [# Gibbs steps=2] elapsed = 112.92955mn
    Epoch[8730] : PPL = 1033.75 [# Gibbs steps=2] elapsed = 113.08586mn
    Epoch[8740] : PPL = 1559.78 [# Gibbs steps=2] elapsed = 113.24582mn
    Epoch[8750] : PPL = 1237.54 [# Gibbs steps=2] elapsed = 113.40704mn
    Epoch[8760] : PPL = 864.07 [# Gibbs steps=2] elapsed = 113.56745mn
    Epoch[8770] : PPL = 1378.83 [# Gibbs steps=2] elapsed = 113.72618mn
    Epoch[8780] : PPL = 1393.37 [# Gibbs steps=2] elapsed = 113.88827mn
    Epoch[8790] : PPL = 1463.84 [# Gibbs steps=2] elapsed = 114.04894mn
    Epoch[8800] : PPL = 1289.29 [# Gibbs steps=2] elapsed = 114.20876mn
    Epoch[8810] : PPL = 1269.63 [# Gibbs steps=2] elapsed = 114.36676mn
    Epoch[8820] : PPL = 1577.60 [# Gibbs steps=2] elapsed = 114.52661mn
    Epoch[8830] : PPL = 1447.30 [# Gibbs steps=2] elapsed = 114.69011mn
    Epoch[8840] : PPL = 995.63 [# Gibbs steps=2] elapsed = 114.85279mn
    Epoch[8850] : PPL = 1446.37 [# Gibbs steps=2] elapsed = 115.01726mn
    Epoch[8860] : PPL = 1003.15 [# Gibbs steps=2] elapsed = 115.17410mn
    Epoch[8870] : PPL = 1309.50 [# Gibbs steps=2] elapsed = 115.33483mn
    Epoch[8880] : PPL = 1302.08 [# Gibbs steps=2] elapsed = 115.49736mn
    Epoch[8890] : PPL = 1564.96 [# Gibbs steps=2] elapsed = 115.65708mn
    Epoch[8900] : PPL = 1457.36 [# Gibbs steps=2] elapsed = 115.81964mn
    Epoch[8910] : PPL = 1634.91 [# Gibbs steps=2] elapsed = 115.98120mn
    Epoch[8920] : PPL = 1214.98 [# Gibbs steps=2] elapsed = 116.14426mn
    Epoch[8930] : PPL = 1057.00 [# Gibbs steps=2] elapsed = 116.30508mn
    Epoch[8940] : PPL = 1535.23 [# Gibbs steps=2] elapsed = 116.46968mn
    Epoch[8950] : PPL = 1487.62 [# Gibbs steps=2] elapsed = 116.63327mn
    Epoch[8960] : PPL = 1370.13 [# Gibbs steps=2] elapsed = 116.79770mn
    Epoch[8970] : PPL = 1556.26 [# Gibbs steps=2] elapsed = 116.95991mn
    Epoch[8980] : PPL = 1511.94 [# Gibbs steps=2] elapsed = 117.12099mn
    Epoch[8990] : PPL = 1580.04 [# Gibbs steps=2] elapsed = 117.28518mn
    Epoch[9000] : PPL = 1259.49 [# Gibbs steps=2] elapsed = 117.62479mn
    Epoch[9010] : PPL = 1146.72 [# Gibbs steps=2] elapsed = 117.78571mn
    Epoch[9020] : PPL = 1415.87 [# Gibbs steps=2] elapsed = 117.94522mn
    Epoch[9030] : PPL = 1078.10 [# Gibbs steps=2] elapsed = 118.10412mn
    Epoch[9040] : PPL = 1510.49 [# Gibbs steps=2] elapsed = 118.26971mn
    Epoch[9050] : PPL = 1424.26 [# Gibbs steps=2] elapsed = 118.43180mn
    Epoch[9060] : PPL = 844.00 [# Gibbs steps=2] elapsed = 118.58897mn
    Epoch[9070] : PPL = 1727.14 [# Gibbs steps=2] elapsed = 118.75193mn
    Epoch[9080] : PPL = 1436.39 [# Gibbs steps=2] elapsed = 118.91265mn
    Epoch[9090] : PPL = 1570.88 [# Gibbs steps=2] elapsed = 119.07389mn
    Epoch[9100] : PPL = 1659.38 [# Gibbs steps=2] elapsed = 119.23772mn
    Epoch[9110] : PPL = 1417.54 [# Gibbs steps=2] elapsed = 119.40045mn
    Epoch[9120] : PPL = 864.71 [# Gibbs steps=2] elapsed = 119.56393mn
    Epoch[9130] : PPL = 936.80 [# Gibbs steps=2] elapsed = 119.72412mn
    Epoch[9140] : PPL = 1626.84 [# Gibbs steps=2] elapsed = 119.88985mn
    Epoch[9150] : PPL = 1270.09 [# Gibbs steps=2] elapsed = 120.05317mn
    Epoch[9160] : PPL = 1294.72 [# Gibbs steps=2] elapsed = 120.21516mn
    Epoch[9170] : PPL = 1573.99 [# Gibbs steps=2] elapsed = 120.38050mn
    Epoch[9180] : PPL = 1341.25 [# Gibbs steps=2] elapsed = 120.54002mn
    Epoch[9190] : PPL = 1201.13 [# Gibbs steps=2] elapsed = 120.70038mn
    Epoch[9200] : PPL = 1565.14 [# Gibbs steps=2] elapsed = 120.86051mn
    Epoch[9210] : PPL = 1408.84 [# Gibbs steps=2] elapsed = 121.02066mn
    Epoch[9220] : PPL = 1603.80 [# Gibbs steps=2] elapsed = 121.18002mn
    Epoch[9230] : PPL = 1488.74 [# Gibbs steps=2] elapsed = 121.34249mn
    Epoch[9240] : PPL = 1563.34 [# Gibbs steps=2] elapsed = 121.50163mn
    Epoch[9250] : PPL = 1193.62 [# Gibbs steps=2] elapsed = 121.66292mn
    Epoch[9260] : PPL = 1460.50 [# Gibbs steps=2] elapsed = 121.82400mn
    Epoch[9270] : PPL = 1613.70 [# Gibbs steps=2] elapsed = 121.98802mn
    Epoch[9280] : PPL = 931.30 [# Gibbs steps=2] elapsed = 122.14541mn
    Epoch[9290] : PPL = 1063.09 [# Gibbs steps=2] elapsed = 122.30406mn
    Epoch[9300] : PPL = 1051.38 [# Gibbs steps=2] elapsed = 122.46785mn
    Epoch[9310] : PPL = 1425.77 [# Gibbs steps=2] elapsed = 122.63210mn
    Epoch[9320] : PPL = 1501.09 [# Gibbs steps=2] elapsed = 122.79374mn
    Epoch[9330] : PPL = 1609.20 [# Gibbs steps=2] elapsed = 122.95323mn
    Epoch[9340] : PPL = 922.68 [# Gibbs steps=2] elapsed = 123.11353mn
    Epoch[9350] : PPL = 1133.11 [# Gibbs steps=2] elapsed = 123.27623mn
    Epoch[9360] : PPL = 1459.34 [# Gibbs steps=2] elapsed = 123.43800mn
    Epoch[9370] : PPL = 1420.21 [# Gibbs steps=2] elapsed = 123.60079mn
    Epoch[9380] : PPL = 1351.81 [# Gibbs steps=2] elapsed = 123.76525mn
    Epoch[9390] : PPL = 1269.39 [# Gibbs steps=2] elapsed = 123.92512mn
    Epoch[9400] : PPL = 1422.19 [# Gibbs steps=2] elapsed = 124.08365mn
    Epoch[9410] : PPL = 1518.35 [# Gibbs steps=2] elapsed = 124.24774mn
    Epoch[9420] : PPL = 1658.81 [# Gibbs steps=2] elapsed = 124.40994mn
    Epoch[9430] : PPL = 1314.27 [# Gibbs steps=2] elapsed = 124.56957mn
    Epoch[9440] : PPL = 1505.55 [# Gibbs steps=2] elapsed = 124.73140mn
    Epoch[9450] : PPL = 1695.10 [# Gibbs steps=2] elapsed = 124.89601mn
    Epoch[9460] : PPL = 1313.75 [# Gibbs steps=2] elapsed = 125.05860mn
    Epoch[9470] : PPL = 1440.04 [# Gibbs steps=2] elapsed = 125.22167mn
    Epoch[9480] : PPL = 1274.77 [# Gibbs steps=2] elapsed = 125.38434mn
    Epoch[9490] : PPL = 1495.59 [# Gibbs steps=2] elapsed = 125.54870mn
    Epoch[9500] : PPL = 1302.19 [# Gibbs steps=2] elapsed = 125.88948mn
    Epoch[9510] : PPL = 1130.07 [# Gibbs steps=2] elapsed = 126.05143mn
    Epoch[9520] : PPL = 1515.51 [# Gibbs steps=2] elapsed = 126.21679mn
    Epoch[9530] : PPL = 1453.18 [# Gibbs steps=2] elapsed = 126.38083mn
    Epoch[9540] : PPL = 1138.53 [# Gibbs steps=2] elapsed = 126.54051mn
    Epoch[9550] : PPL = 1317.46 [# Gibbs steps=2] elapsed = 126.70308mn
    Epoch[9560] : PPL = 1451.47 [# Gibbs steps=2] elapsed = 126.86787mn
    Epoch[9570] : PPL = 903.79 [# Gibbs steps=2] elapsed = 127.03015mn
    Epoch[9580] : PPL = 1607.30 [# Gibbs steps=2] elapsed = 127.18964mn
    Epoch[9590] : PPL = 1494.31 [# Gibbs steps=2] elapsed = 127.35477mn
    Epoch[9600] : PPL = 2006.54 [# Gibbs steps=2] elapsed = 127.51649mn
    Epoch[9610] : PPL = 1366.34 [# Gibbs steps=2] elapsed = 127.67570mn
    Epoch[9620] : PPL = 1269.48 [# Gibbs steps=2] elapsed = 127.83752mn
    Epoch[9630] : PPL = 1271.25 [# Gibbs steps=2] elapsed = 127.99690mn
    Epoch[9640] : PPL = 1097.02 [# Gibbs steps=2] elapsed = 128.16006mn
    Epoch[9650] : PPL = 1396.90 [# Gibbs steps=2] elapsed = 128.31989mn
    Epoch[9660] : PPL = 1652.39 [# Gibbs steps=2] elapsed = 128.48327mn
    Epoch[9670] : PPL = 1292.83 [# Gibbs steps=2] elapsed = 128.64778mn
    Epoch[9680] : PPL = 1451.44 [# Gibbs steps=2] elapsed = 128.81015mn
    Epoch[9690] : PPL = 1436.81 [# Gibbs steps=2] elapsed = 128.97017mn
    Epoch[9700] : PPL = 1386.22 [# Gibbs steps=2] elapsed = 129.13087mn
    Epoch[9710] : PPL = 1171.89 [# Gibbs steps=2] elapsed = 129.29234mn
    Epoch[9720] : PPL = 1481.13 [# Gibbs steps=2] elapsed = 129.45622mn
    Epoch[9730] : PPL = 1439.95 [# Gibbs steps=2] elapsed = 129.61374mn
    Epoch[9740] : PPL = 1410.30 [# Gibbs steps=2] elapsed = 129.77441mn
    Epoch[9750] : PPL = 1447.12 [# Gibbs steps=2] elapsed = 129.93611mn
    Epoch[9760] : PPL = 1429.63 [# Gibbs steps=2] elapsed = 130.09466mn
    Epoch[9770] : PPL = 1576.74 [# Gibbs steps=2] elapsed = 130.25908mn
    Epoch[9780] : PPL = 1157.92 [# Gibbs steps=2] elapsed = 130.42027mn
    Epoch[9790] : PPL = 1009.83 [# Gibbs steps=2] elapsed = 130.57857mn
    Epoch[9800] : PPL = 896.56 [# Gibbs steps=2] elapsed = 130.74002mn
    Epoch[9810] : PPL = 1159.36 [# Gibbs steps=2] elapsed = 130.89829mn
    Epoch[9820] : PPL = 1494.97 [# Gibbs steps=2] elapsed = 131.06150mn
    Epoch[9830] : PPL = 1428.03 [# Gibbs steps=2] elapsed = 131.22310mn
    Epoch[9840] : PPL = 1028.91 [# Gibbs steps=2] elapsed = 131.38298mn
    Epoch[9850] : PPL = 806.88 [# Gibbs steps=2] elapsed = 131.54165mn
    Epoch[9860] : PPL = 1441.59 [# Gibbs steps=2] elapsed = 131.70636mn
    Epoch[9870] : PPL = 1362.95 [# Gibbs steps=2] elapsed = 131.86552mn
    Epoch[9880] : PPL = 1481.18 [# Gibbs steps=2] elapsed = 132.02615mn
    Epoch[9890] : PPL = 1652.08 [# Gibbs steps=2] elapsed = 132.18751mn
    Epoch[9900] : PPL = 1371.41 [# Gibbs steps=2] elapsed = 132.34906mn
    Epoch[9910] : PPL = 1177.08 [# Gibbs steps=2] elapsed = 132.51222mn
    Epoch[9920] : PPL = 837.82 [# Gibbs steps=2] elapsed = 132.67092mn
    Epoch[9930] : PPL = 991.28 [# Gibbs steps=2] elapsed = 132.82973mn
    Epoch[9940] : PPL = 1250.32 [# Gibbs steps=2] elapsed = 132.99485mn
    Epoch[9950] : PPL = 1326.47 [# Gibbs steps=2] elapsed = 133.15756mn
    Epoch[9960] : PPL = 1442.57 [# Gibbs steps=2] elapsed = 133.31634mn
    Epoch[9970] : PPL = 1567.04 [# Gibbs steps=2] elapsed = 133.47756mn
    Epoch[9980] : PPL = 1311.59 [# Gibbs steps=2] elapsed = 133.64257mn
    Epoch[9990] : PPL = 1734.08 [# Gibbs steps=2] elapsed = 133.80560mn
    Epoch[10000] : PPL = 1493.16 [# Gibbs steps=3] elapsed = 134.15300mn
    Epoch[10010] : PPL = 1484.10 [# Gibbs steps=3] elapsed = 134.37649mn
    Epoch[10020] : PPL = 1381.42 [# Gibbs steps=3] elapsed = 134.59985mn
    Epoch[10030] : PPL = 1478.97 [# Gibbs steps=3] elapsed = 134.82582mn
    Epoch[10040] : PPL = 1533.75 [# Gibbs steps=3] elapsed = 135.05145mn
    Epoch[10050] : PPL = 1199.44 [# Gibbs steps=3] elapsed = 135.27381mn
    Epoch[10060] : PPL = 1333.39 [# Gibbs steps=3] elapsed = 135.50105mn
    Epoch[10070] : PPL = 975.48 [# Gibbs steps=3] elapsed = 135.72671mn
    Epoch[10080] : PPL = 1507.34 [# Gibbs steps=3] elapsed = 135.95308mn
    Epoch[10090] : PPL = 1625.88 [# Gibbs steps=3] elapsed = 136.17944mn
    Epoch[10100] : PPL = 1382.98 [# Gibbs steps=3] elapsed = 136.40727mn
    Epoch[10110] : PPL = 1432.00 [# Gibbs steps=3] elapsed = 136.63234mn
    Epoch[10120] : PPL = 1435.23 [# Gibbs steps=3] elapsed = 136.85326mn
    Epoch[10130] : PPL = 1604.79 [# Gibbs steps=3] elapsed = 137.07531mn
    Epoch[10140] : PPL = 1528.33 [# Gibbs steps=3] elapsed = 137.30013mn
    Epoch[10150] : PPL = 1384.00 [# Gibbs steps=3] elapsed = 137.52535mn
    Epoch[10160] : PPL = 1640.79 [# Gibbs steps=3] elapsed = 137.74535mn
    Epoch[10170] : PPL = 1531.34 [# Gibbs steps=3] elapsed = 137.97115mn
    Epoch[10180] : PPL = 1454.15 [# Gibbs steps=3] elapsed = 138.19635mn
    Epoch[10190] : PPL = 1474.03 [# Gibbs steps=3] elapsed = 138.42158mn
    Epoch[10200] : PPL = 1520.17 [# Gibbs steps=3] elapsed = 138.64989mn
    Epoch[10210] : PPL = 1453.45 [# Gibbs steps=3] elapsed = 138.87543mn
    Epoch[10220] : PPL = 1489.68 [# Gibbs steps=3] elapsed = 139.10272mn
    Epoch[10230] : PPL = 1453.00 [# Gibbs steps=3] elapsed = 139.32844mn
    Epoch[10240] : PPL = 1429.76 [# Gibbs steps=3] elapsed = 139.55405mn
    Epoch[10250] : PPL = 927.82 [# Gibbs steps=3] elapsed = 139.77896mn
    Epoch[10260] : PPL = 1518.16 [# Gibbs steps=3] elapsed = 140.00176mn
    Epoch[10270] : PPL = 1196.68 [# Gibbs steps=3] elapsed = 140.22704mn
    Epoch[10280] : PPL = 1039.34 [# Gibbs steps=3] elapsed = 140.45140mn
    Epoch[10290] : PPL = 877.15 [# Gibbs steps=3] elapsed = 140.67536mn
    Epoch[10300] : PPL = 1298.74 [# Gibbs steps=3] elapsed = 140.89855mn
    Epoch[10310] : PPL = 1259.84 [# Gibbs steps=3] elapsed = 141.12354mn
    Epoch[10320] : PPL = 1129.55 [# Gibbs steps=3] elapsed = 141.34471mn
    Epoch[10330] : PPL = 1474.05 [# Gibbs steps=3] elapsed = 141.56617mn
    Epoch[10340] : PPL = 1134.79 [# Gibbs steps=3] elapsed = 141.79251mn
    Epoch[10350] : PPL = 1310.68 [# Gibbs steps=3] elapsed = 142.01992mn
    Epoch[10360] : PPL = 1008.49 [# Gibbs steps=3] elapsed = 142.24232mn
    Epoch[10370] : PPL = 1119.50 [# Gibbs steps=3] elapsed = 142.46947mn
    Epoch[10380] : PPL = 1211.07 [# Gibbs steps=3] elapsed = 142.69396mn
    Epoch[10390] : PPL = 1061.05 [# Gibbs steps=3] elapsed = 142.91948mn
    Epoch[10400] : PPL = 1000.76 [# Gibbs steps=3] elapsed = 143.14453mn
    Epoch[10410] : PPL = 1516.06 [# Gibbs steps=3] elapsed = 143.37224mn
    Epoch[10420] : PPL = 1574.75 [# Gibbs steps=3] elapsed = 143.59576mn
    Epoch[10430] : PPL = 1327.17 [# Gibbs steps=3] elapsed = 143.82110mn
    Epoch[10440] : PPL = 1379.32 [# Gibbs steps=3] elapsed = 144.04608mn
    Epoch[10450] : PPL = 1463.82 [# Gibbs steps=3] elapsed = 144.27463mn
    Epoch[10460] : PPL = 2010.45 [# Gibbs steps=3] elapsed = 144.50059mn
    Epoch[10470] : PPL = 1151.30 [# Gibbs steps=3] elapsed = 144.72584mn
    Epoch[10480] : PPL = 2030.04 [# Gibbs steps=3] elapsed = 144.95027mn
    Epoch[10490] : PPL = 1280.67 [# Gibbs steps=3] elapsed = 145.17458mn
    Epoch[10500] : PPL = 1542.86 [# Gibbs steps=3] elapsed = 145.57629mn
    Epoch[10510] : PPL = 1173.50 [# Gibbs steps=3] elapsed = 145.80426mn
    Epoch[10520] : PPL = 1247.34 [# Gibbs steps=3] elapsed = 146.02697mn
    Epoch[10530] : PPL = 1371.55 [# Gibbs steps=3] elapsed = 146.25060mn
    Epoch[10540] : PPL = 1612.38 [# Gibbs steps=3] elapsed = 146.47701mn
    Epoch[10550] : PPL = 1413.86 [# Gibbs steps=3] elapsed = 146.70137mn
    Epoch[10560] : PPL = 1000.63 [# Gibbs steps=3] elapsed = 146.92511mn
    Epoch[10570] : PPL = 842.70 [# Gibbs steps=3] elapsed = 147.15198mn
    Epoch[10580] : PPL = 1290.90 [# Gibbs steps=3] elapsed = 147.37613mn
    Epoch[10590] : PPL = 1403.84 [# Gibbs steps=3] elapsed = 147.60592mn
    Epoch[10600] : PPL = 1277.86 [# Gibbs steps=3] elapsed = 147.83078mn
    Epoch[10610] : PPL = 1588.71 [# Gibbs steps=3] elapsed = 148.05315mn
    Epoch[10620] : PPL = 1307.98 [# Gibbs steps=3] elapsed = 148.27895mn
    Epoch[10630] : PPL = 908.81 [# Gibbs steps=3] elapsed = 148.50491mn
    Epoch[10640] : PPL = 1286.22 [# Gibbs steps=3] elapsed = 148.72710mn
    Epoch[10650] : PPL = 1447.94 [# Gibbs steps=3] elapsed = 148.95319mn
    Epoch[10660] : PPL = 1648.56 [# Gibbs steps=3] elapsed = 149.17597mn
    Epoch[10670] : PPL = 1101.16 [# Gibbs steps=3] elapsed = 149.39602mn
    Epoch[10680] : PPL = 1563.60 [# Gibbs steps=3] elapsed = 149.62353mn
    Epoch[10690] : PPL = 1506.01 [# Gibbs steps=3] elapsed = 149.84773mn
    Epoch[10700] : PPL = 1504.87 [# Gibbs steps=3] elapsed = 150.07153mn
    Epoch[10710] : PPL = 1201.93 [# Gibbs steps=3] elapsed = 150.30335mn
    Epoch[10720] : PPL = 1226.07 [# Gibbs steps=3] elapsed = 150.53138mn
    Epoch[10730] : PPL = 1322.63 [# Gibbs steps=3] elapsed = 150.76171mn
    Epoch[10740] : PPL = 1492.81 [# Gibbs steps=3] elapsed = 150.98877mn
    Epoch[10750] : PPL = 1088.08 [# Gibbs steps=3] elapsed = 151.21140mn
    Epoch[10760] : PPL = 1579.88 [# Gibbs steps=3] elapsed = 151.43713mn
    Epoch[10770] : PPL = 1177.78 [# Gibbs steps=3] elapsed = 151.66222mn
    Epoch[10780] : PPL = 1352.41 [# Gibbs steps=3] elapsed = 151.89166mn
    Epoch[10790] : PPL = 1285.27 [# Gibbs steps=3] elapsed = 152.11894mn
    Epoch[10800] : PPL = 1471.71 [# Gibbs steps=3] elapsed = 152.34119mn
    Epoch[10810] : PPL = 1429.14 [# Gibbs steps=3] elapsed = 152.56532mn
    Epoch[10820] : PPL = 1786.36 [# Gibbs steps=3] elapsed = 152.79441mn
    Epoch[10830] : PPL = 1271.37 [# Gibbs steps=3] elapsed = 153.01755mn
    Epoch[10840] : PPL = 1155.98 [# Gibbs steps=3] elapsed = 153.24601mn
    Epoch[10850] : PPL = 847.60 [# Gibbs steps=3] elapsed = 153.46602mn
    Epoch[10860] : PPL = 1159.12 [# Gibbs steps=3] elapsed = 153.68914mn
    Epoch[10870] : PPL = 933.93 [# Gibbs steps=3] elapsed = 153.91677mn
    Epoch[10880] : PPL = 1613.70 [# Gibbs steps=3] elapsed = 154.13999mn
    Epoch[10890] : PPL = 1396.99 [# Gibbs steps=3] elapsed = 154.36600mn
    Epoch[10900] : PPL = 1433.08 [# Gibbs steps=3] elapsed = 154.59573mn
    Epoch[10910] : PPL = 1644.68 [# Gibbs steps=3] elapsed = 154.82155mn
    Epoch[10920] : PPL = 1464.80 [# Gibbs steps=3] elapsed = 155.04584mn
    Epoch[10930] : PPL = 1643.17 [# Gibbs steps=3] elapsed = 155.27051mn
    Epoch[10940] : PPL = 866.87 [# Gibbs steps=3] elapsed = 155.49217mn
    Epoch[10950] : PPL = 1355.43 [# Gibbs steps=3] elapsed = 155.71594mn
    Epoch[10960] : PPL = 1642.25 [# Gibbs steps=3] elapsed = 155.94133mn
    Epoch[10970] : PPL = 1507.69 [# Gibbs steps=3] elapsed = 156.16506mn
    Epoch[10980] : PPL = 1229.93 [# Gibbs steps=3] elapsed = 156.39479mn
    Epoch[10990] : PPL = 1538.85 [# Gibbs steps=3] elapsed = 156.61932mn
    Epoch[11000] : PPL = 1293.25 [# Gibbs steps=3] elapsed = 157.02161mn
    Epoch[11010] : PPL = 884.41 [# Gibbs steps=3] elapsed = 157.24636mn
    Epoch[11020] : PPL = 1925.69 [# Gibbs steps=3] elapsed = 157.46924mn
    Epoch[11030] : PPL = 1060.45 [# Gibbs steps=3] elapsed = 157.69230mn
    Epoch[11040] : PPL = 1347.89 [# Gibbs steps=3] elapsed = 157.91813mn
    Epoch[11050] : PPL = 898.77 [# Gibbs steps=3] elapsed = 158.14433mn
    Epoch[11060] : PPL = 1941.64 [# Gibbs steps=3] elapsed = 158.37217mn
    Epoch[11070] : PPL = 890.40 [# Gibbs steps=3] elapsed = 158.59537mn
    Epoch[11080] : PPL = 1536.11 [# Gibbs steps=3] elapsed = 158.82304mn
    Epoch[11090] : PPL = 1377.79 [# Gibbs steps=3] elapsed = 159.04729mn
    Epoch[11100] : PPL = 856.80 [# Gibbs steps=3] elapsed = 159.26983mn
    Epoch[11110] : PPL = 1382.49 [# Gibbs steps=3] elapsed = 159.49531mn
    Epoch[11120] : PPL = 1615.24 [# Gibbs steps=3] elapsed = 159.72320mn
    Epoch[11130] : PPL = 1589.30 [# Gibbs steps=3] elapsed = 159.94871mn
    Epoch[11140] : PPL = 989.00 [# Gibbs steps=3] elapsed = 160.17350mn
    Epoch[11150] : PPL = 1358.36 [# Gibbs steps=3] elapsed = 160.39833mn
    Epoch[11160] : PPL = 1605.89 [# Gibbs steps=3] elapsed = 160.62522mn
    Epoch[11170] : PPL = 1111.50 [# Gibbs steps=3] elapsed = 160.85062mn
    Epoch[11180] : PPL = 1305.33 [# Gibbs steps=3] elapsed = 161.07739mn
    Epoch[11190] : PPL = 1138.95 [# Gibbs steps=3] elapsed = 161.30227mn
    Epoch[11200] : PPL = 1251.31 [# Gibbs steps=3] elapsed = 161.53111mn
    Epoch[11210] : PPL = 1484.79 [# Gibbs steps=3] elapsed = 161.75607mn
    Epoch[11220] : PPL = 1362.89 [# Gibbs steps=3] elapsed = 161.98075mn
    Epoch[11230] : PPL = 1290.55 [# Gibbs steps=3] elapsed = 162.20123mn
    Epoch[11240] : PPL = 1486.39 [# Gibbs steps=3] elapsed = 162.42627mn
    Epoch[11250] : PPL = 1319.05 [# Gibbs steps=3] elapsed = 162.64732mn
    Epoch[11260] : PPL = 1132.89 [# Gibbs steps=3] elapsed = 162.87324mn
    Epoch[11270] : PPL = 1472.14 [# Gibbs steps=3] elapsed = 163.10000mn
    Epoch[11280] : PPL = 1479.06 [# Gibbs steps=3] elapsed = 163.32072mn
    Epoch[11290] : PPL = 1538.77 [# Gibbs steps=3] elapsed = 163.54445mn
    Epoch[11300] : PPL = 1240.51 [# Gibbs steps=3] elapsed = 163.77062mn
    Epoch[11310] : PPL = 957.70 [# Gibbs steps=3] elapsed = 163.99687mn
    Epoch[11320] : PPL = 1423.04 [# Gibbs steps=3] elapsed = 164.22187mn
    Epoch[11330] : PPL = 1220.25 [# Gibbs steps=3] elapsed = 164.45061mn
    Epoch[11340] : PPL = 1310.62 [# Gibbs steps=3] elapsed = 164.67501mn
    Epoch[11350] : PPL = 1288.12 [# Gibbs steps=3] elapsed = 164.89846mn
    Epoch[11360] : PPL = 1658.42 [# Gibbs steps=3] elapsed = 165.12274mn
    Epoch[11370] : PPL = 1026.57 [# Gibbs steps=3] elapsed = 165.34672mn
    Epoch[11380] : PPL = 1419.32 [# Gibbs steps=3] elapsed = 165.56704mn
    Epoch[11390] : PPL = 1414.88 [# Gibbs steps=3] elapsed = 165.78931mn
    Epoch[11400] : PPL = 1326.25 [# Gibbs steps=3] elapsed = 166.01866mn
    Epoch[11410] : PPL = 1602.52 [# Gibbs steps=3] elapsed = 166.24304mn
    Epoch[11420] : PPL = 1278.68 [# Gibbs steps=3] elapsed = 166.46673mn
    Epoch[11430] : PPL = 1134.23 [# Gibbs steps=3] elapsed = 166.68983mn
    Epoch[11440] : PPL = 1441.15 [# Gibbs steps=3] elapsed = 166.91584mn
    Epoch[11450] : PPL = 1431.65 [# Gibbs steps=3] elapsed = 167.14520mn
    Epoch[11460] : PPL = 1525.83 [# Gibbs steps=3] elapsed = 167.37266mn
    Epoch[11470] : PPL = 1428.25 [# Gibbs steps=3] elapsed = 167.60278mn
    Epoch[11480] : PPL = 1540.62 [# Gibbs steps=3] elapsed = 167.83057mn
    Epoch[11490] : PPL = 1379.82 [# Gibbs steps=3] elapsed = 168.05077mn
    Epoch[11500] : PPL = 1366.29 [# Gibbs steps=3] elapsed = 168.44884mn
    Epoch[11510] : PPL = 876.75 [# Gibbs steps=3] elapsed = 168.67318mn
    Epoch[11520] : PPL = 1479.09 [# Gibbs steps=3] elapsed = 168.89754mn
    Epoch[11530] : PPL = 1373.00 [# Gibbs steps=3] elapsed = 169.12066mn
    Epoch[11540] : PPL = 918.74 [# Gibbs steps=3] elapsed = 169.34561mn
    Epoch[11550] : PPL = 1017.91 [# Gibbs steps=3] elapsed = 169.56821mn
    Epoch[11560] : PPL = 1439.19 [# Gibbs steps=3] elapsed = 169.79667mn
    Epoch[11570] : PPL = 1325.02 [# Gibbs steps=3] elapsed = 170.01981mn
    Epoch[11580] : PPL = 1330.90 [# Gibbs steps=3] elapsed = 170.23913mn
    Epoch[11590] : PPL = 1328.48 [# Gibbs steps=3] elapsed = 170.46153mn
    Epoch[11600] : PPL = 839.77 [# Gibbs steps=3] elapsed = 170.68261mn
    Epoch[11610] : PPL = 1528.69 [# Gibbs steps=3] elapsed = 170.90976mn
    Epoch[11620] : PPL = 2012.00 [# Gibbs steps=3] elapsed = 171.13540mn
    Epoch[11630] : PPL = 1356.77 [# Gibbs steps=3] elapsed = 171.36290mn
    Epoch[11640] : PPL = 1497.86 [# Gibbs steps=3] elapsed = 171.58921mn
    Epoch[11650] : PPL = 1222.22 [# Gibbs steps=3] elapsed = 171.81532mn
    Epoch[11660] : PPL = 1340.23 [# Gibbs steps=3] elapsed = 172.04313mn
    Epoch[11670] : PPL = 1395.44 [# Gibbs steps=3] elapsed = 172.26769mn
    Epoch[11680] : PPL = 971.96 [# Gibbs steps=3] elapsed = 172.48992mn
    Epoch[11690] : PPL = 1544.56 [# Gibbs steps=3] elapsed = 172.71606mn
    Epoch[11700] : PPL = 1306.38 [# Gibbs steps=3] elapsed = 172.94525mn
    Epoch[11710] : PPL = 1189.73 [# Gibbs steps=3] elapsed = 173.17199mn
    Epoch[11720] : PPL = 1159.45 [# Gibbs steps=3] elapsed = 173.39965mn
    Epoch[11730] : PPL = 1606.83 [# Gibbs steps=3] elapsed = 173.62415mn
    Epoch[11740] : PPL = 1285.87 [# Gibbs steps=3] elapsed = 173.84843mn
    Epoch[11750] : PPL = 958.05 [# Gibbs steps=3] elapsed = 174.07444mn
    Epoch[11760] : PPL = 1525.50 [# Gibbs steps=3] elapsed = 174.30338mn
    Epoch[11770] : PPL = 1272.17 [# Gibbs steps=3] elapsed = 174.52532mn
    Epoch[11780] : PPL = 1528.44 [# Gibbs steps=3] elapsed = 174.74923mn
    Epoch[11790] : PPL = 874.57 [# Gibbs steps=3] elapsed = 174.97159mn
    Epoch[11800] : PPL = 1452.56 [# Gibbs steps=3] elapsed = 175.19914mn
    Epoch[11810] : PPL = 1227.40 [# Gibbs steps=3] elapsed = 175.42376mn
    Epoch[11820] : PPL = 1548.62 [# Gibbs steps=3] elapsed = 175.64731mn
    Epoch[11830] : PPL = 1390.14 [# Gibbs steps=3] elapsed = 175.87339mn
    Epoch[11840] : PPL = 1433.23 [# Gibbs steps=3] elapsed = 176.09696mn
    Epoch[11850] : PPL = 1019.54 [# Gibbs steps=3] elapsed = 176.32136mn
    Epoch[11860] : PPL = 1023.07 [# Gibbs steps=3] elapsed = 176.54307mn
    Epoch[11870] : PPL = 1390.49 [# Gibbs steps=3] elapsed = 176.77134mn
    Epoch[11880] : PPL = 882.53 [# Gibbs steps=3] elapsed = 176.99715mn
    Epoch[11890] : PPL = 1287.05 [# Gibbs steps=3] elapsed = 177.22452mn
    Epoch[11900] : PPL = 1399.04 [# Gibbs steps=3] elapsed = 177.45444mn
    Epoch[11910] : PPL = 1350.73 [# Gibbs steps=3] elapsed = 177.67671mn
    Epoch[11920] : PPL = 1405.82 [# Gibbs steps=3] elapsed = 177.89854mn
    Epoch[11930] : PPL = 1254.94 [# Gibbs steps=3] elapsed = 178.12446mn
    Epoch[11940] : PPL = 1033.88 [# Gibbs steps=3] elapsed = 178.34803mn
    Epoch[11950] : PPL = 1533.65 [# Gibbs steps=3] elapsed = 178.57371mn
    Epoch[11960] : PPL = 1345.20 [# Gibbs steps=3] elapsed = 178.80139mn
    Epoch[11970] : PPL = 1262.84 [# Gibbs steps=3] elapsed = 179.02662mn
    Epoch[11980] : PPL = 1512.23 [# Gibbs steps=3] elapsed = 179.25402mn
    Epoch[11990] : PPL = 1262.90 [# Gibbs steps=3] elapsed = 179.47670mn
    Epoch[12000] : PPL = 1186.17 [# Gibbs steps=3] elapsed = 179.88539mn
    Epoch[12010] : PPL = 928.13 [# Gibbs steps=3] elapsed = 180.11052mn
    Epoch[12020] : PPL = 1349.55 [# Gibbs steps=3] elapsed = 180.33720mn
    Epoch[12030] : PPL = 1590.33 [# Gibbs steps=3] elapsed = 180.56147mn
    Epoch[12040] : PPL = 1324.97 [# Gibbs steps=3] elapsed = 180.78705mn
    Epoch[12050] : PPL = 1428.98 [# Gibbs steps=3] elapsed = 181.01555mn
    Epoch[12060] : PPL = 1570.46 [# Gibbs steps=3] elapsed = 181.24203mn
    Epoch[12070] : PPL = 1549.20 [# Gibbs steps=3] elapsed = 181.47059mn
    Epoch[12080] : PPL = 1457.21 [# Gibbs steps=3] elapsed = 181.69756mn
    Epoch[12090] : PPL = 1118.26 [# Gibbs steps=3] elapsed = 181.92587mn
    Epoch[12100] : PPL = 1459.39 [# Gibbs steps=3] elapsed = 182.15358mn
    Epoch[12110] : PPL = 928.12 [# Gibbs steps=3] elapsed = 182.37818mn
    Epoch[12120] : PPL = 1472.57 [# Gibbs steps=3] elapsed = 182.60266mn
    Epoch[12130] : PPL = 1358.98 [# Gibbs steps=3] elapsed = 182.83057mn
    Epoch[12140] : PPL = 1277.62 [# Gibbs steps=3] elapsed = 183.05634mn
    Epoch[12150] : PPL = 1253.92 [# Gibbs steps=3] elapsed = 183.27839mn
    Epoch[12160] : PPL = 1044.77 [# Gibbs steps=3] elapsed = 183.49882mn
    Epoch[12170] : PPL = 1081.97 [# Gibbs steps=3] elapsed = 183.72246mn
    Epoch[12180] : PPL = 1085.05 [# Gibbs steps=3] elapsed = 183.94397mn
    Epoch[12190] : PPL = 1437.01 [# Gibbs steps=3] elapsed = 184.16921mn
    Epoch[12200] : PPL = 1546.03 [# Gibbs steps=3] elapsed = 184.39343mn
    Epoch[12210] : PPL = 1422.22 [# Gibbs steps=3] elapsed = 184.61555mn
    Epoch[12220] : PPL = 1328.26 [# Gibbs steps=3] elapsed = 184.83703mn
    Epoch[12230] : PPL = 1093.67 [# Gibbs steps=3] elapsed = 185.05988mn
    Epoch[12240] : PPL = 1383.27 [# Gibbs steps=3] elapsed = 185.28357mn
    Epoch[12250] : PPL = 1330.36 [# Gibbs steps=3] elapsed = 185.50492mn
    Epoch[12260] : PPL = 1216.44 [# Gibbs steps=3] elapsed = 185.72800mn
    Epoch[12270] : PPL = 1949.49 [# Gibbs steps=3] elapsed = 185.95015mn
    Epoch[12280] : PPL = 1252.79 [# Gibbs steps=3] elapsed = 186.17347mn
    Epoch[12290] : PPL = 1692.60 [# Gibbs steps=3] elapsed = 186.39752mn
    Epoch[12300] : PPL = 1378.89 [# Gibbs steps=3] elapsed = 186.62323mn
    Epoch[12310] : PPL = 897.10 [# Gibbs steps=3] elapsed = 186.84548mn
    Epoch[12320] : PPL = 1300.17 [# Gibbs steps=3] elapsed = 187.07127mn
    Epoch[12330] : PPL = 1650.75 [# Gibbs steps=3] elapsed = 187.29768mn
    Epoch[12340] : PPL = 1372.51 [# Gibbs steps=3] elapsed = 187.51677mn
    Epoch[12350] : PPL = 1371.21 [# Gibbs steps=3] elapsed = 187.74423mn
    Epoch[12360] : PPL = 1013.06 [# Gibbs steps=3] elapsed = 187.96645mn
    Epoch[12370] : PPL = 1467.51 [# Gibbs steps=3] elapsed = 188.19434mn
    Epoch[12380] : PPL = 1624.66 [# Gibbs steps=3] elapsed = 188.41445mn
    Epoch[12390] : PPL = 1123.77 [# Gibbs steps=3] elapsed = 188.63579mn
    Epoch[12400] : PPL = 1165.90 [# Gibbs steps=3] elapsed = 188.86182mn
    Epoch[12410] : PPL = 1189.86 [# Gibbs steps=3] elapsed = 189.08712mn
    Epoch[12420] : PPL = 1393.11 [# Gibbs steps=3] elapsed = 189.31414mn
    Epoch[12430] : PPL = 1367.47 [# Gibbs steps=3] elapsed = 189.54035mn
    Epoch[12440] : PPL = 1436.24 [# Gibbs steps=3] elapsed = 189.76663mn
    Epoch[12450] : PPL = 1322.38 [# Gibbs steps=3] elapsed = 189.99072mn
    Epoch[12460] : PPL = 897.66 [# Gibbs steps=3] elapsed = 190.21465mn
    Epoch[12470] : PPL = 1588.71 [# Gibbs steps=3] elapsed = 190.44461mn
    Epoch[12480] : PPL = 1419.11 [# Gibbs steps=3] elapsed = 190.67133mn
    Epoch[12490] : PPL = 1452.18 [# Gibbs steps=3] elapsed = 190.89884mn
    Epoch[12500] : PPL = 1445.95 [# Gibbs steps=3] elapsed = 191.29851mn
    Epoch[12510] : PPL = 1168.18 [# Gibbs steps=3] elapsed = 191.52452mn
    Epoch[12520] : PPL = 1494.15 [# Gibbs steps=3] elapsed = 191.75018mn
    Epoch[12530] : PPL = 1199.79 [# Gibbs steps=3] elapsed = 191.97799mn
    Epoch[12540] : PPL = 1711.66 [# Gibbs steps=3] elapsed = 192.20166mn
    Epoch[12550] : PPL = 1570.78 [# Gibbs steps=3] elapsed = 192.42602mn
    Epoch[12560] : PPL = 1507.07 [# Gibbs steps=3] elapsed = 192.65167mn
    Epoch[12570] : PPL = 942.83 [# Gibbs steps=3] elapsed = 192.87730mn
    Epoch[12580] : PPL = 1203.94 [# Gibbs steps=3] elapsed = 193.10444mn
    Epoch[12590] : PPL = 1427.43 [# Gibbs steps=3] elapsed = 193.33104mn
    Epoch[12600] : PPL = 841.80 [# Gibbs steps=3] elapsed = 193.55456mn
    Epoch[12610] : PPL = 1540.08 [# Gibbs steps=3] elapsed = 193.77865mn
    Epoch[12620] : PPL = 1353.49 [# Gibbs steps=3] elapsed = 194.00089mn
    Epoch[12630] : PPL = 1562.52 [# Gibbs steps=3] elapsed = 194.22966mn
    Epoch[12640] : PPL = 1454.02 [# Gibbs steps=3] elapsed = 194.45289mn
    Epoch[12650] : PPL = 1358.32 [# Gibbs steps=3] elapsed = 194.67482mn
    Epoch[12660] : PPL = 1418.59 [# Gibbs steps=3] elapsed = 194.89762mn
    Epoch[12670] : PPL = 1589.77 [# Gibbs steps=3] elapsed = 195.12538mn
    Epoch[12680] : PPL = 1154.94 [# Gibbs steps=3] elapsed = 195.34651mn
    Epoch[12690] : PPL = 1315.67 [# Gibbs steps=3] elapsed = 195.57341mn
    Epoch[12700] : PPL = 1484.80 [# Gibbs steps=3] elapsed = 195.79662mn
    Epoch[12710] : PPL = 1569.40 [# Gibbs steps=3] elapsed = 196.02382mn
    Epoch[12720] : PPL = 860.33 [# Gibbs steps=3] elapsed = 196.24574mn
    Epoch[12730] : PPL = 1528.91 [# Gibbs steps=3] elapsed = 196.46634mn
    Epoch[12740] : PPL = 1532.01 [# Gibbs steps=3] elapsed = 196.69390mn
    Epoch[12750] : PPL = 1447.11 [# Gibbs steps=3] elapsed = 196.92264mn
    Epoch[12760] : PPL = 1601.80 [# Gibbs steps=3] elapsed = 197.15220mn
    Epoch[12770] : PPL = 1562.65 [# Gibbs steps=3] elapsed = 197.37577mn
    Epoch[12780] : PPL = 920.29 [# Gibbs steps=3] elapsed = 197.60056mn
    Epoch[12790] : PPL = 1258.28 [# Gibbs steps=3] elapsed = 197.82547mn
    Epoch[12800] : PPL = 1532.27 [# Gibbs steps=3] elapsed = 198.04969mn
    Epoch[12810] : PPL = 1066.93 [# Gibbs steps=3] elapsed = 198.27130mn
    Epoch[12820] : PPL = 1089.43 [# Gibbs steps=3] elapsed = 198.49379mn
    Epoch[12830] : PPL = 1261.73 [# Gibbs steps=3] elapsed = 198.71849mn
    Epoch[12840] : PPL = 941.71 [# Gibbs steps=3] elapsed = 198.94098mn
    Epoch[12850] : PPL = 1989.72 [# Gibbs steps=3] elapsed = 199.16484mn
    Epoch[12860] : PPL = 1287.28 [# Gibbs steps=3] elapsed = 199.39034mn
    Epoch[12870] : PPL = 1593.61 [# Gibbs steps=3] elapsed = 199.61247mn
    Epoch[12880] : PPL = 1348.98 [# Gibbs steps=3] elapsed = 199.83611mn
    Epoch[12890] : PPL = 1584.93 [# Gibbs steps=3] elapsed = 200.06379mn
    Epoch[12900] : PPL = 1451.95 [# Gibbs steps=3] elapsed = 200.29022mn
    Epoch[12910] : PPL = 1803.38 [# Gibbs steps=3] elapsed = 200.51540mn
    Epoch[12920] : PPL = 1403.51 [# Gibbs steps=3] elapsed = 200.74008mn
    Epoch[12930] : PPL = 869.74 [# Gibbs steps=3] elapsed = 200.96107mn
    Epoch[12940] : PPL = 1514.93 [# Gibbs steps=3] elapsed = 201.18623mn
    Epoch[12950] : PPL = 1745.09 [# Gibbs steps=3] elapsed = 201.41486mn
    Epoch[12960] : PPL = 1492.65 [# Gibbs steps=3] elapsed = 201.64099mn
    Epoch[12970] : PPL = 1476.72 [# Gibbs steps=3] elapsed = 201.86596mn
    Epoch[12980] : PPL = 1485.18 [# Gibbs steps=3] elapsed = 202.09188mn
    Epoch[12990] : PPL = 1166.66 [# Gibbs steps=3] elapsed = 202.31532mn
    Epoch[13000] : PPL = 1239.48 [# Gibbs steps=3] elapsed = 202.72118mn
    Epoch[13010] : PPL = 1651.02 [# Gibbs steps=3] elapsed = 202.94556mn
    Epoch[13020] : PPL = 1424.36 [# Gibbs steps=3] elapsed = 203.16906mn
    Epoch[13030] : PPL = 832.89 [# Gibbs steps=3] elapsed = 203.39067mn
    Epoch[13040] : PPL = 1549.28 [# Gibbs steps=3] elapsed = 203.61908mn
    Epoch[13050] : PPL = 1121.22 [# Gibbs steps=3] elapsed = 203.83797mn
    Epoch[13060] : PPL = 1550.37 [# Gibbs steps=3] elapsed = 204.06288mn
    Epoch[13070] : PPL = 1555.88 [# Gibbs steps=3] elapsed = 204.28889mn
    Epoch[13080] : PPL = 840.40 [# Gibbs steps=3] elapsed = 204.51166mn
    Epoch[13090] : PPL = 1519.33 [# Gibbs steps=3] elapsed = 204.73760mn
    Epoch[13100] : PPL = 1434.88 [# Gibbs steps=3] elapsed = 204.96195mn
    Epoch[13110] : PPL = 1368.55 [# Gibbs steps=3] elapsed = 205.18532mn
    Epoch[13120] : PPL = 916.12 [# Gibbs steps=3] elapsed = 205.40838mn
    Epoch[13130] : PPL = 1294.54 [# Gibbs steps=3] elapsed = 205.63657mn
    Epoch[13140] : PPL = 1023.05 [# Gibbs steps=3] elapsed = 205.86069mn
    Epoch[13150] : PPL = 925.09 [# Gibbs steps=3] elapsed = 206.08363mn
    Epoch[13160] : PPL = 851.30 [# Gibbs steps=3] elapsed = 206.31006mn
    Epoch[13170] : PPL = 1333.89 [# Gibbs steps=3] elapsed = 206.53777mn
    Epoch[13180] : PPL = 1344.34 [# Gibbs steps=3] elapsed = 206.76229mn
    Epoch[13190] : PPL = 1199.73 [# Gibbs steps=3] elapsed = 206.99094mn
    Epoch[13200] : PPL = 1132.30 [# Gibbs steps=3] elapsed = 207.21431mn
    Epoch[13210] : PPL = 1272.34 [# Gibbs steps=3] elapsed = 207.43883mn
    Epoch[13220] : PPL = 1115.57 [# Gibbs steps=3] elapsed = 207.66438mn
    Epoch[13230] : PPL = 1328.05 [# Gibbs steps=3] elapsed = 207.89301mn
    Epoch[13240] : PPL = 1201.82 [# Gibbs steps=3] elapsed = 208.11806mn
    Epoch[13250] : PPL = 1319.92 [# Gibbs steps=3] elapsed = 208.34367mn
    Epoch[13260] : PPL = 1045.37 [# Gibbs steps=3] elapsed = 208.57120mn
    Epoch[13270] : PPL = 1493.72 [# Gibbs steps=3] elapsed = 208.79390mn
    Epoch[13280] : PPL = 932.70 [# Gibbs steps=3] elapsed = 209.01863mn
    Epoch[13290] : PPL = 1393.94 [# Gibbs steps=3] elapsed = 209.24306mn
    Epoch[13300] : PPL = 1631.12 [# Gibbs steps=3] elapsed = 209.46778mn
    Epoch[13310] : PPL = 1479.96 [# Gibbs steps=3] elapsed = 209.69103mn
    Epoch[13320] : PPL = 1571.51 [# Gibbs steps=3] elapsed = 209.91612mn
    Epoch[13330] : PPL = 1415.64 [# Gibbs steps=3] elapsed = 210.13854mn
    Epoch[13340] : PPL = 915.29 [# Gibbs steps=3] elapsed = 210.35978mn
    Epoch[13350] : PPL = 1315.58 [# Gibbs steps=3] elapsed = 210.58832mn
    Epoch[13360] : PPL = 1308.47 [# Gibbs steps=3] elapsed = 210.81352mn
    Epoch[13370] : PPL = 956.14 [# Gibbs steps=3] elapsed = 211.03776mn
    Epoch[13380] : PPL = 1465.01 [# Gibbs steps=3] elapsed = 211.26103mn
    Epoch[13390] : PPL = 1131.62 [# Gibbs steps=3] elapsed = 211.48741mn
    Epoch[13400] : PPL = 1284.50 [# Gibbs steps=3] elapsed = 211.71596mn
    Epoch[13410] : PPL = 1700.38 [# Gibbs steps=3] elapsed = 211.94091mn
    Epoch[13420] : PPL = 1110.69 [# Gibbs steps=3] elapsed = 212.16416mn
    Epoch[13430] : PPL = 1299.39 [# Gibbs steps=3] elapsed = 212.39030mn
    Epoch[13440] : PPL = 1151.14 [# Gibbs steps=3] elapsed = 212.61509mn
    Epoch[13450] : PPL = 1421.85 [# Gibbs steps=3] elapsed = 212.83823mn
    Epoch[13460] : PPL = 1048.41 [# Gibbs steps=3] elapsed = 213.06289mn
    Epoch[13470] : PPL = 1462.41 [# Gibbs steps=3] elapsed = 213.28949mn
    Epoch[13480] : PPL = 1288.05 [# Gibbs steps=3] elapsed = 213.50949mn
    Epoch[13490] : PPL = 1617.69 [# Gibbs steps=3] elapsed = 213.73409mn
    Epoch[13500] : PPL = 1230.70 [# Gibbs steps=3] elapsed = 214.13643mn
    Epoch[13510] : PPL = 1311.15 [# Gibbs steps=3] elapsed = 214.35855mn
    Epoch[13520] : PPL = 1531.26 [# Gibbs steps=3] elapsed = 214.58911mn
    Epoch[13530] : PPL = 1219.91 [# Gibbs steps=3] elapsed = 214.81646mn
    Epoch[13540] : PPL = 1539.66 [# Gibbs steps=3] elapsed = 215.04370mn
    Epoch[13550] : PPL = 914.67 [# Gibbs steps=3] elapsed = 215.26819mn
    Epoch[13560] : PPL = 1397.07 [# Gibbs steps=3] elapsed = 215.49441mn
    Epoch[13570] : PPL = 1535.11 [# Gibbs steps=3] elapsed = 215.72200mn
    Epoch[13580] : PPL = 1163.71 [# Gibbs steps=3] elapsed = 215.94386mn
    Epoch[13590] : PPL = 1194.15 [# Gibbs steps=3] elapsed = 216.17130mn
    Epoch[13600] : PPL = 1532.87 [# Gibbs steps=3] elapsed = 216.39442mn
    Epoch[13610] : PPL = 935.45 [# Gibbs steps=3] elapsed = 216.61865mn
    Epoch[13620] : PPL = 1565.23 [# Gibbs steps=3] elapsed = 216.84589mn
    Epoch[13630] : PPL = 1218.87 [# Gibbs steps=3] elapsed = 217.07138mn
    Epoch[13640] : PPL = 1333.97 [# Gibbs steps=3] elapsed = 217.29878mn
    Epoch[13650] : PPL = 1457.85 [# Gibbs steps=3] elapsed = 217.52392mn
    Epoch[13660] : PPL = 1479.58 [# Gibbs steps=3] elapsed = 217.74998mn
    Epoch[13670] : PPL = 1330.37 [# Gibbs steps=3] elapsed = 217.97352mn
    Epoch[13680] : PPL = 1372.07 [# Gibbs steps=3] elapsed = 218.20005mn
    Epoch[13690] : PPL = 1421.65 [# Gibbs steps=3] elapsed = 218.42927mn
    Epoch[13700] : PPL = 1321.01 [# Gibbs steps=3] elapsed = 218.65496mn
    Epoch[13710] : PPL = 1272.15 [# Gibbs steps=3] elapsed = 218.87777mn
    Epoch[13720] : PPL = 1495.17 [# Gibbs steps=3] elapsed = 219.10410mn
    Epoch[13730] : PPL = 1369.61 [# Gibbs steps=3] elapsed = 219.33114mn
    Epoch[13740] : PPL = 1216.45 [# Gibbs steps=3] elapsed = 219.55525mn
    Epoch[13750] : PPL = 1520.19 [# Gibbs steps=3] elapsed = 219.77697mn
    Epoch[13760] : PPL = 1041.53 [# Gibbs steps=3] elapsed = 219.99852mn
    Epoch[13770] : PPL = 1304.45 [# Gibbs steps=3] elapsed = 220.22415mn
    Epoch[13780] : PPL = 993.99 [# Gibbs steps=3] elapsed = 220.44949mn
    Epoch[13790] : PPL = 1463.87 [# Gibbs steps=3] elapsed = 220.67479mn
    Epoch[13800] : PPL = 1721.42 [# Gibbs steps=3] elapsed = 220.90053mn
    Epoch[13810] : PPL = 1575.95 [# Gibbs steps=3] elapsed = 221.12917mn
    Epoch[13820] : PPL = 917.32 [# Gibbs steps=3] elapsed = 221.35582mn
    Epoch[13830] : PPL = 1481.29 [# Gibbs steps=3] elapsed = 221.58136mn
    Epoch[13840] : PPL = 1407.95 [# Gibbs steps=3] elapsed = 221.80491mn
    Epoch[13850] : PPL = 920.43 [# Gibbs steps=3] elapsed = 222.02874mn
    Epoch[13860] : PPL = 1563.82 [# Gibbs steps=3] elapsed = 222.25570mn
    Epoch[13870] : PPL = 1739.50 [# Gibbs steps=3] elapsed = 222.47971mn
    Epoch[13880] : PPL = 938.08 [# Gibbs steps=3] elapsed = 222.70454mn
    Epoch[13890] : PPL = 890.18 [# Gibbs steps=3] elapsed = 222.92638mn
    Epoch[13900] : PPL = 1321.51 [# Gibbs steps=3] elapsed = 223.15192mn
    Epoch[13910] : PPL = 852.11 [# Gibbs steps=3] elapsed = 223.37932mn
    Epoch[13920] : PPL = 1601.96 [# Gibbs steps=3] elapsed = 223.60210mn
    Epoch[13930] : PPL = 1262.14 [# Gibbs steps=3] elapsed = 223.82334mn
    Epoch[13940] : PPL = 1064.52 [# Gibbs steps=3] elapsed = 224.04789mn
    Epoch[13950] : PPL = 1552.17 [# Gibbs steps=3] elapsed = 224.27536mn
    Epoch[13960] : PPL = 989.63 [# Gibbs steps=3] elapsed = 224.49952mn
    Epoch[13970] : PPL = 1412.40 [# Gibbs steps=3] elapsed = 224.72026mn
    Epoch[13980] : PPL = 1101.54 [# Gibbs steps=3] elapsed = 224.94360mn
    Epoch[13990] : PPL = 828.97 [# Gibbs steps=3] elapsed = 225.16539mn
    Epoch[14000] : PPL = 1316.21 [# Gibbs steps=3] elapsed = 225.57016mn
    Epoch[14010] : PPL = 1528.44 [# Gibbs steps=3] elapsed = 225.79631mn
    Epoch[14020] : PPL = 1553.00 [# Gibbs steps=3] elapsed = 226.02117mn
    Epoch[14030] : PPL = 1092.09 [# Gibbs steps=3] elapsed = 226.24198mn
    Epoch[14040] : PPL = 1094.83 [# Gibbs steps=3] elapsed = 226.46720mn
    Epoch[14050] : PPL = 1268.52 [# Gibbs steps=3] elapsed = 226.69385mn
    Epoch[14060] : PPL = 1522.90 [# Gibbs steps=3] elapsed = 226.92031mn
    Epoch[14070] : PPL = 1021.06 [# Gibbs steps=3] elapsed = 227.14689mn
    Epoch[14080] : PPL = 1524.20 [# Gibbs steps=3] elapsed = 227.37459mn
    Epoch[14090] : PPL = 1737.55 [# Gibbs steps=3] elapsed = 227.60217mn
    Epoch[14100] : PPL = 1140.09 [# Gibbs steps=3] elapsed = 227.82814mn
    Epoch[14110] : PPL = 1195.34 [# Gibbs steps=3] elapsed = 228.05666mn
    Epoch[14120] : PPL = 1278.41 [# Gibbs steps=3] elapsed = 228.28169mn
    Epoch[14130] : PPL = 1510.08 [# Gibbs steps=3] elapsed = 228.50665mn
    Epoch[14140] : PPL = 1302.26 [# Gibbs steps=3] elapsed = 228.72978mn
    Epoch[14150] : PPL = 1658.67 [# Gibbs steps=3] elapsed = 228.95492mn
    Epoch[14160] : PPL = 1378.43 [# Gibbs steps=3] elapsed = 229.17672mn
    Epoch[14170] : PPL = 1419.14 [# Gibbs steps=3] elapsed = 229.40281mn
    Epoch[14180] : PPL = 1509.77 [# Gibbs steps=3] elapsed = 229.62672mn
    Epoch[14190] : PPL = 1544.43 [# Gibbs steps=3] elapsed = 229.85146mn
    Epoch[14200] : PPL = 1433.83 [# Gibbs steps=3] elapsed = 230.07794mn
    Epoch[14210] : PPL = 1549.81 [# Gibbs steps=3] elapsed = 230.30198mn
    Epoch[14220] : PPL = 1378.56 [# Gibbs steps=3] elapsed = 230.52541mn
    Epoch[14230] : PPL = 1568.36 [# Gibbs steps=3] elapsed = 230.75168mn
    Epoch[14240] : PPL = 1686.56 [# Gibbs steps=3] elapsed = 230.97361mn
    Epoch[14250] : PPL = 1573.20 [# Gibbs steps=3] elapsed = 231.19646mn
    Epoch[14260] : PPL = 1547.02 [# Gibbs steps=3] elapsed = 231.42173mn
    Epoch[14270] : PPL = 1604.13 [# Gibbs steps=3] elapsed = 231.64720mn
    Epoch[14280] : PPL = 1123.44 [# Gibbs steps=3] elapsed = 231.87355mn
    Epoch[14290] : PPL = 1481.98 [# Gibbs steps=3] elapsed = 232.10011mn
    Epoch[14300] : PPL = 1352.14 [# Gibbs steps=3] elapsed = 232.32628mn
    Epoch[14310] : PPL = 1636.42 [# Gibbs steps=3] elapsed = 232.54837mn
    Epoch[14320] : PPL = 898.13 [# Gibbs steps=3] elapsed = 232.77377mn
    Epoch[14330] : PPL = 1249.25 [# Gibbs steps=3] elapsed = 233.00073mn
    Epoch[14340] : PPL = 1288.92 [# Gibbs steps=3] elapsed = 233.22495mn
    Epoch[14350] : PPL = 1371.19 [# Gibbs steps=3] elapsed = 233.45019mn
    Epoch[14360] : PPL = 1426.05 [# Gibbs steps=3] elapsed = 233.67511mn
    Epoch[14370] : PPL = 1457.94 [# Gibbs steps=3] elapsed = 233.89771mn
    Epoch[14380] : PPL = 1543.40 [# Gibbs steps=3] elapsed = 234.12179mn
    Epoch[14390] : PPL = 1065.90 [# Gibbs steps=3] elapsed = 234.34591mn
    Epoch[14400] : PPL = 1439.15 [# Gibbs steps=3] elapsed = 234.57199mn
    Epoch[14410] : PPL = 1582.93 [# Gibbs steps=3] elapsed = 234.79743mn
    Epoch[14420] : PPL = 900.88 [# Gibbs steps=3] elapsed = 235.02251mn
    Epoch[14430] : PPL = 915.04 [# Gibbs steps=3] elapsed = 235.24442mn
    Epoch[14440] : PPL = 1407.71 [# Gibbs steps=3] elapsed = 235.46965mn
    Epoch[14450] : PPL = 1594.15 [# Gibbs steps=3] elapsed = 235.69345mn
    Epoch[14460] : PPL = 1477.03 [# Gibbs steps=3] elapsed = 235.91485mn
    Epoch[14470] : PPL = 1365.64 [# Gibbs steps=3] elapsed = 236.14155mn
    Epoch[14480] : PPL = 1620.08 [# Gibbs steps=3] elapsed = 236.37041mn
    Epoch[14490] : PPL = 3512.71 [# Gibbs steps=3] elapsed = 236.60043mn
    Epoch[14500] : PPL = 1590.99 [# Gibbs steps=3] elapsed = 237.00383mn
    Epoch[14510] : PPL = 1305.73 [# Gibbs steps=3] elapsed = 237.22410mn
    Epoch[14520] : PPL = 1426.29 [# Gibbs steps=3] elapsed = 237.44579mn
    Epoch[14530] : PPL = 1583.99 [# Gibbs steps=3] elapsed = 237.66587mn
    Epoch[14540] : PPL = 1227.78 [# Gibbs steps=3] elapsed = 237.89036mn
    Epoch[14550] : PPL = 857.85 [# Gibbs steps=3] elapsed = 238.11415mn
    Epoch[14560] : PPL = 1488.94 [# Gibbs steps=3] elapsed = 238.33798mn
    Epoch[14570] : PPL = 1543.13 [# Gibbs steps=3] elapsed = 238.56484mn
    Epoch[14580] : PPL = 1476.16 [# Gibbs steps=3] elapsed = 238.79148mn
    Epoch[14590] : PPL = 1138.28 [# Gibbs steps=3] elapsed = 239.01511mn
    Epoch[14600] : PPL = 1227.12 [# Gibbs steps=3] elapsed = 239.23534mn
    Epoch[14610] : PPL = 1070.72 [# Gibbs steps=3] elapsed = 239.46007mn
    Epoch[14620] : PPL = 1505.34 [# Gibbs steps=3] elapsed = 239.68528mn
    Epoch[14630] : PPL = 1256.92 [# Gibbs steps=3] elapsed = 239.91046mn
    Epoch[14640] : PPL = 1418.14 [# Gibbs steps=3] elapsed = 240.13429mn
    Epoch[14650] : PPL = 1502.67 [# Gibbs steps=3] elapsed = 240.36118mn
    Epoch[14660] : PPL = 1306.94 [# Gibbs steps=3] elapsed = 240.58311mn
    Epoch[14670] : PPL = 1528.59 [# Gibbs steps=3] elapsed = 240.81042mn
    Epoch[14680] : PPL = 1546.48 [# Gibbs steps=3] elapsed = 241.03742mn
    Epoch[14690] : PPL = 1022.02 [# Gibbs steps=3] elapsed = 241.26196mn
    Epoch[14700] : PPL = 1327.00 [# Gibbs steps=3] elapsed = 241.48460mn
    Epoch[14710] : PPL = 1251.99 [# Gibbs steps=3] elapsed = 241.70671mn
    Epoch[14720] : PPL = 1515.85 [# Gibbs steps=3] elapsed = 241.93173mn
    Epoch[14730] : PPL = 1109.10 [# Gibbs steps=3] elapsed = 242.15630mn
    Epoch[14740] : PPL = 1572.66 [# Gibbs steps=3] elapsed = 242.38176mn
    Epoch[14750] : PPL = 1435.40 [# Gibbs steps=3] elapsed = 242.60900mn
    Epoch[14760] : PPL = 880.75 [# Gibbs steps=3] elapsed = 242.83598mn
    Epoch[14770] : PPL = 1622.22 [# Gibbs steps=3] elapsed = 243.06169mn
    Epoch[14780] : PPL = 1569.35 [# Gibbs steps=3] elapsed = 243.28497mn
    Epoch[14790] : PPL = 1906.01 [# Gibbs steps=3] elapsed = 243.50781mn
    Epoch[14800] : PPL = 1278.03 [# Gibbs steps=3] elapsed = 243.73253mn
    Epoch[14810] : PPL = 1290.30 [# Gibbs steps=3] elapsed = 243.95915mn
    Epoch[14820] : PPL = 928.82 [# Gibbs steps=3] elapsed = 244.18372mn
    Epoch[14830] : PPL = 1210.71 [# Gibbs steps=3] elapsed = 244.40955mn
    Epoch[14840] : PPL = 1503.04 [# Gibbs steps=3] elapsed = 244.63632mn
    Epoch[14850] : PPL = 1105.19 [# Gibbs steps=3] elapsed = 244.86243mn
    Epoch[14860] : PPL = 1197.83 [# Gibbs steps=3] elapsed = 245.08954mn
    Epoch[14870] : PPL = 1188.34 [# Gibbs steps=3] elapsed = 245.31341mn
    Epoch[14880] : PPL = 1457.59 [# Gibbs steps=3] elapsed = 245.53937mn
    Epoch[14890] : PPL = 1355.45 [# Gibbs steps=3] elapsed = 245.76297mn
    Epoch[14900] : PPL = 1597.74 [# Gibbs steps=3] elapsed = 245.99375mn
    Epoch[14910] : PPL = 1509.24 [# Gibbs steps=3] elapsed = 246.21614mn
    Epoch[14920] : PPL = 1108.53 [# Gibbs steps=3] elapsed = 246.43937mn
    Epoch[14930] : PPL = 1459.17 [# Gibbs steps=3] elapsed = 246.66259mn
    Epoch[14940] : PPL = 972.05 [# Gibbs steps=3] elapsed = 246.88844mn
    Epoch[14950] : PPL = 1249.37 [# Gibbs steps=3] elapsed = 247.11365mn
    Epoch[14960] : PPL = 1135.28 [# Gibbs steps=3] elapsed = 247.33768mn
    Epoch[14970] : PPL = 1311.96 [# Gibbs steps=3] elapsed = 247.56147mn
    Epoch[14980] : PPL = 1359.19 [# Gibbs steps=3] elapsed = 247.78505mn
    Epoch[14990] : PPL = 1085.29 [# Gibbs steps=3] elapsed = 248.00843mn
    Epoch[15000] : PPL = 1531.67 [# Gibbs steps=4] elapsed = 248.41737mn
    Epoch[15010] : PPL = 1532.50 [# Gibbs steps=4] elapsed = 248.70451mn
    Epoch[15020] : PPL = 868.32 [# Gibbs steps=4] elapsed = 248.99122mn
    Epoch[15030] : PPL = 1297.91 [# Gibbs steps=4] elapsed = 249.27795mn
    Epoch[15040] : PPL = 1613.58 [# Gibbs steps=4] elapsed = 249.56798mn
    Epoch[15050] : PPL = 1331.16 [# Gibbs steps=4] elapsed = 249.85601mn
    Epoch[15060] : PPL = 898.06 [# Gibbs steps=4] elapsed = 250.14291mn
    Epoch[15070] : PPL = 1388.65 [# Gibbs steps=4] elapsed = 250.43197mn
    Epoch[15080] : PPL = 1188.99 [# Gibbs steps=4] elapsed = 250.72272mn
    Epoch[15090] : PPL = 1575.34 [# Gibbs steps=4] elapsed = 251.00708mn
    Epoch[15100] : PPL = 1592.51 [# Gibbs steps=4] elapsed = 251.29849mn
    Epoch[15110] : PPL = 1509.18 [# Gibbs steps=4] elapsed = 251.58984mn
    Epoch[15120] : PPL = 1541.72 [# Gibbs steps=4] elapsed = 251.88193mn
    Epoch[15130] : PPL = 1493.55 [# Gibbs steps=4] elapsed = 252.17053mn
    Epoch[15140] : PPL = 1516.66 [# Gibbs steps=4] elapsed = 252.46244mn
    Epoch[15150] : PPL = 1369.48 [# Gibbs steps=4] elapsed = 252.75005mn
    Epoch[15160] : PPL = 1440.54 [# Gibbs steps=4] elapsed = 253.04162mn
    Epoch[15170] : PPL = 1317.55 [# Gibbs steps=4] elapsed = 253.32923mn
    Epoch[15180] : PPL = 1293.16 [# Gibbs steps=4] elapsed = 253.61766mn
    Epoch[15190] : PPL = 1373.63 [# Gibbs steps=4] elapsed = 253.90673mn
    Epoch[15200] : PPL = 886.40 [# Gibbs steps=4] elapsed = 254.19522mn
    Epoch[15210] : PPL = 1637.31 [# Gibbs steps=4] elapsed = 254.48292mn
    Epoch[15220] : PPL = 1394.39 [# Gibbs steps=4] elapsed = 254.77170mn
    Epoch[15230] : PPL = 1359.17 [# Gibbs steps=4] elapsed = 255.06182mn
    Epoch[15240] : PPL = 1563.47 [# Gibbs steps=4] elapsed = 255.35278mn
    Epoch[15250] : PPL = 1324.16 [# Gibbs steps=4] elapsed = 255.64153mn
    Epoch[15260] : PPL = 1207.05 [# Gibbs steps=4] elapsed = 255.93328mn
    Epoch[15270] : PPL = 1535.66 [# Gibbs steps=4] elapsed = 256.22057mn
    Epoch[15280] : PPL = 1307.27 [# Gibbs steps=4] elapsed = 256.50912mn
    Epoch[15290] : PPL = 1604.62 [# Gibbs steps=4] elapsed = 256.79483mn
    Epoch[15300] : PPL = 1107.71 [# Gibbs steps=4] elapsed = 257.08463mn
    Epoch[15310] : PPL = 1478.49 [# Gibbs steps=4] elapsed = 257.37472mn
    Epoch[15320] : PPL = 1709.11 [# Gibbs steps=4] elapsed = 257.66377mn
    Epoch[15330] : PPL = 889.64 [# Gibbs steps=4] elapsed = 257.95189mn
    Epoch[15340] : PPL = 1487.81 [# Gibbs steps=4] elapsed = 258.24299mn
    Epoch[15350] : PPL = 1261.76 [# Gibbs steps=4] elapsed = 258.52894mn
    Epoch[15360] : PPL = 1552.95 [# Gibbs steps=4] elapsed = 258.81175mn
    Epoch[15370] : PPL = 1288.97 [# Gibbs steps=4] elapsed = 259.10118mn
    Epoch[15380] : PPL = 1445.09 [# Gibbs steps=4] elapsed = 259.39118mn
    Epoch[15390] : PPL = 1410.60 [# Gibbs steps=4] elapsed = 259.68093mn
    Epoch[15400] : PPL = 1119.84 [# Gibbs steps=4] elapsed = 259.96803mn
    Epoch[15410] : PPL = 1590.98 [# Gibbs steps=4] elapsed = 260.25806mn
    Epoch[15420] : PPL = 1497.45 [# Gibbs steps=4] elapsed = 260.54607mn
    Epoch[15430] : PPL = 1448.20 [# Gibbs steps=4] elapsed = 260.83216mn
    Epoch[15440] : PPL = 1375.81 [# Gibbs steps=4] elapsed = 261.11781mn
    Epoch[15450] : PPL = 1571.76 [# Gibbs steps=4] elapsed = 261.40736mn
    Epoch[15460] : PPL = 1497.04 [# Gibbs steps=4] elapsed = 261.69590mn
    Epoch[15470] : PPL = 1103.59 [# Gibbs steps=4] elapsed = 261.98293mn
    Epoch[15480] : PPL = 942.73 [# Gibbs steps=4] elapsed = 262.27260mn
    Epoch[15490] : PPL = 1551.88 [# Gibbs steps=4] elapsed = 262.56283mn
    Epoch[15500] : PPL = 1367.36 [# Gibbs steps=4] elapsed = 263.02936mn
    Epoch[15510] : PPL = 1078.17 [# Gibbs steps=4] elapsed = 263.31973mn
    Epoch[15520] : PPL = 1303.49 [# Gibbs steps=4] elapsed = 263.61093mn
    Epoch[15530] : PPL = 1399.19 [# Gibbs steps=4] elapsed = 263.89875mn
    Epoch[15540] : PPL = 1386.60 [# Gibbs steps=4] elapsed = 264.18806mn
    Epoch[15550] : PPL = 1745.70 [# Gibbs steps=4] elapsed = 264.47756mn
    Epoch[15560] : PPL = 1575.42 [# Gibbs steps=4] elapsed = 264.76663mn
    Epoch[15570] : PPL = 1173.14 [# Gibbs steps=4] elapsed = 265.05519mn
    Epoch[15580] : PPL = 1065.85 [# Gibbs steps=4] elapsed = 265.34366mn
    Epoch[15590] : PPL = 1408.86 [# Gibbs steps=4] elapsed = 265.63868mn
    Epoch[15600] : PPL = 1555.74 [# Gibbs steps=4] elapsed = 265.92693mn
    Epoch[15610] : PPL = 1529.93 [# Gibbs steps=4] elapsed = 266.21711mn
    Epoch[15620] : PPL = 964.59 [# Gibbs steps=4] elapsed = 266.50556mn
    Epoch[15630] : PPL = 869.50 [# Gibbs steps=4] elapsed = 266.79468mn
    Epoch[15640] : PPL = 820.64 [# Gibbs steps=4] elapsed = 267.08323mn
    Epoch[15650] : PPL = 1290.22 [# Gibbs steps=4] elapsed = 267.37360mn
    Epoch[15660] : PPL = 1246.75 [# Gibbs steps=4] elapsed = 267.66064mn
    Epoch[15670] : PPL = 1408.83 [# Gibbs steps=4] elapsed = 267.95040mn
    Epoch[15680] : PPL = 882.62 [# Gibbs steps=4] elapsed = 268.24284mn
    Epoch[15690] : PPL = 928.55 [# Gibbs steps=4] elapsed = 268.52862mn
    Epoch[15700] : PPL = 1542.66 [# Gibbs steps=4] elapsed = 268.81556mn
    Epoch[15710] : PPL = 1643.30 [# Gibbs steps=4] elapsed = 269.10652mn
    Epoch[15720] : PPL = 1338.94 [# Gibbs steps=4] elapsed = 269.39703mn
    Epoch[15730] : PPL = 1542.77 [# Gibbs steps=4] elapsed = 269.68559mn
    Epoch[15740] : PPL = 1329.94 [# Gibbs steps=4] elapsed = 269.97651mn
    Epoch[15750] : PPL = 1356.14 [# Gibbs steps=4] elapsed = 270.26820mn
    Epoch[15760] : PPL = 1372.98 [# Gibbs steps=4] elapsed = 270.55443mn
    Epoch[15770] : PPL = 1602.75 [# Gibbs steps=4] elapsed = 270.84809mn
    Epoch[15780] : PPL = 1559.37 [# Gibbs steps=4] elapsed = 271.13818mn
    Epoch[15790] : PPL = 1131.12 [# Gibbs steps=4] elapsed = 271.42528mn
    Epoch[15800] : PPL = 1021.49 [# Gibbs steps=4] elapsed = 271.71211mn
    Epoch[15810] : PPL = 872.94 [# Gibbs steps=4] elapsed = 271.99536mn
    Epoch[15820] : PPL = 1434.68 [# Gibbs steps=4] elapsed = 272.28546mn
    Epoch[15830] : PPL = 1097.67 [# Gibbs steps=4] elapsed = 272.57305mn
    Epoch[15840] : PPL = 1532.19 [# Gibbs steps=4] elapsed = 272.86058mn
    Epoch[15850] : PPL = 1428.05 [# Gibbs steps=4] elapsed = 273.14968mn
    Epoch[15860] : PPL = 1340.16 [# Gibbs steps=4] elapsed = 273.43794mn
    Epoch[15870] : PPL = 1505.86 [# Gibbs steps=4] elapsed = 273.72785mn
    Epoch[15880] : PPL = 1625.25 [# Gibbs steps=4] elapsed = 274.01667mn
    Epoch[15890] : PPL = 1556.49 [# Gibbs steps=4] elapsed = 274.30509mn
    Epoch[15900] : PPL = 1322.17 [# Gibbs steps=4] elapsed = 274.59409mn
    Epoch[15910] : PPL = 1415.21 [# Gibbs steps=4] elapsed = 274.88461mn
    Epoch[15920] : PPL = 1303.61 [# Gibbs steps=4] elapsed = 275.17264mn
    Epoch[15930] : PPL = 1644.69 [# Gibbs steps=4] elapsed = 275.46275mn
    Epoch[15940] : PPL = 1358.96 [# Gibbs steps=4] elapsed = 275.75482mn
    Epoch[15950] : PPL = 1169.74 [# Gibbs steps=4] elapsed = 276.04185mn
    Epoch[15960] : PPL = 1116.31 [# Gibbs steps=4] elapsed = 276.32894mn
    Epoch[15970] : PPL = 1574.37 [# Gibbs steps=4] elapsed = 276.61574mn
    Epoch[15980] : PPL = 1281.69 [# Gibbs steps=4] elapsed = 276.90573mn
    Epoch[15990] : PPL = 1091.50 [# Gibbs steps=4] elapsed = 277.19252mn
    Epoch[16000] : PPL = 1166.08 [# Gibbs steps=4] elapsed = 277.65672mn
    Epoch[16010] : PPL = 1323.47 [# Gibbs steps=4] elapsed = 277.94196mn
    Epoch[16020] : PPL = 1260.76 [# Gibbs steps=4] elapsed = 278.23061mn
    Epoch[16030] : PPL = 1457.48 [# Gibbs steps=4] elapsed = 278.51915mn
    Epoch[16040] : PPL = 897.33 [# Gibbs steps=4] elapsed = 278.80676mn
    Epoch[16050] : PPL = 1030.71 [# Gibbs steps=4] elapsed = 279.09448mn
    Epoch[16060] : PPL = 1467.58 [# Gibbs steps=4] elapsed = 279.38613mn
    Epoch[16070] : PPL = 1296.26 [# Gibbs steps=4] elapsed = 279.67673mn
    Epoch[16080] : PPL = 1373.94 [# Gibbs steps=4] elapsed = 279.96530mn
    Epoch[16090] : PPL = 975.19 [# Gibbs steps=4] elapsed = 280.25475mn
    Epoch[16100] : PPL = 1328.95 [# Gibbs steps=4] elapsed = 280.54503mn
    Epoch[16110] : PPL = 1575.06 [# Gibbs steps=4] elapsed = 280.83060mn
    Epoch[16120] : PPL = 2048.34 [# Gibbs steps=4] elapsed = 281.11947mn
    Epoch[16130] : PPL = 1560.62 [# Gibbs steps=4] elapsed = 281.40518mn
    Epoch[16140] : PPL = 988.58 [# Gibbs steps=4] elapsed = 281.69365mn
    Epoch[16150] : PPL = 1983.48 [# Gibbs steps=4] elapsed = 281.98168mn
    Epoch[16160] : PPL = 1552.79 [# Gibbs steps=4] elapsed = 282.26569mn
    Epoch[16170] : PPL = 1751.05 [# Gibbs steps=4] elapsed = 282.55303mn
    Epoch[16180] : PPL = 1361.72 [# Gibbs steps=4] elapsed = 282.84248mn
    Epoch[16190] : PPL = 1339.83 [# Gibbs steps=4] elapsed = 283.13152mn
    Epoch[16200] : PPL = 1055.53 [# Gibbs steps=4] elapsed = 283.42174mn
    Epoch[16210] : PPL = 1493.86 [# Gibbs steps=4] elapsed = 283.71354mn
    Epoch[16220] : PPL = 1055.78 [# Gibbs steps=4] elapsed = 284.00249mn
    Epoch[16230] : PPL = 1592.97 [# Gibbs steps=4] elapsed = 284.29234mn
    Epoch[16240] : PPL = 843.98 [# Gibbs steps=4] elapsed = 284.57919mn
    Epoch[16250] : PPL = 1475.60 [# Gibbs steps=4] elapsed = 284.86395mn
    Epoch[16260] : PPL = 1474.75 [# Gibbs steps=4] elapsed = 285.15235mn
    Epoch[16270] : PPL = 1225.80 [# Gibbs steps=4] elapsed = 285.44091mn
    Epoch[16280] : PPL = 962.60 [# Gibbs steps=4] elapsed = 285.72769mn
    Epoch[16290] : PPL = 1005.69 [# Gibbs steps=4] elapsed = 286.01521mn
    Epoch[16300] : PPL = 1194.56 [# Gibbs steps=4] elapsed = 286.30623mn
    Epoch[16310] : PPL = 1658.41 [# Gibbs steps=4] elapsed = 286.59442mn
    Epoch[16320] : PPL = 1320.95 [# Gibbs steps=4] elapsed = 286.88409mn
    Epoch[16330] : PPL = 1598.50 [# Gibbs steps=4] elapsed = 287.17736mn
    Epoch[16340] : PPL = 1540.65 [# Gibbs steps=4] elapsed = 287.46594mn
    Epoch[16350] : PPL = 1614.64 [# Gibbs steps=4] elapsed = 287.75600mn
    Epoch[16360] : PPL = 973.09 [# Gibbs steps=4] elapsed = 288.04570mn
    Epoch[16370] : PPL = 1015.35 [# Gibbs steps=4] elapsed = 288.33058mn
    Epoch[16380] : PPL = 1056.80 [# Gibbs steps=4] elapsed = 288.61584mn
    Epoch[16390] : PPL = 1185.44 [# Gibbs steps=4] elapsed = 288.90212mn
    Epoch[16400] : PPL = 1550.13 [# Gibbs steps=4] elapsed = 289.19258mn
    Epoch[16410] : PPL = 1297.80 [# Gibbs steps=4] elapsed = 289.48463mn
    Epoch[16420] : PPL = 1306.57 [# Gibbs steps=4] elapsed = 289.77247mn
    Epoch[16430] : PPL = 1548.98 [# Gibbs steps=4] elapsed = 290.06294mn
    Epoch[16440] : PPL = 1620.35 [# Gibbs steps=4] elapsed = 290.35260mn
    Epoch[16450] : PPL = 1435.30 [# Gibbs steps=4] elapsed = 290.63953mn
    Epoch[16460] : PPL = 903.96 [# Gibbs steps=4] elapsed = 290.92815mn
    Epoch[16470] : PPL = 1250.55 [# Gibbs steps=4] elapsed = 291.21714mn
    Epoch[16480] : PPL = 1358.62 [# Gibbs steps=4] elapsed = 291.50817mn
    Epoch[16490] : PPL = 888.17 [# Gibbs steps=4] elapsed = 291.79391mn
    Epoch[16500] : PPL = 1316.30 [# Gibbs steps=4] elapsed = 292.26299mn
    Epoch[16510] : PPL = 1642.08 [# Gibbs steps=4] elapsed = 292.55018mn
    Epoch[16520] : PPL = 1288.66 [# Gibbs steps=4] elapsed = 292.83976mn
    Epoch[16530] : PPL = 1516.01 [# Gibbs steps=4] elapsed = 293.12991mn
    Epoch[16540] : PPL = 1136.06 [# Gibbs steps=4] elapsed = 293.41662mn
    Epoch[16550] : PPL = 1043.34 [# Gibbs steps=4] elapsed = 293.70160mn
    Epoch[16560] : PPL = 1206.51 [# Gibbs steps=4] elapsed = 293.99448mn
    Epoch[16570] : PPL = 1178.93 [# Gibbs steps=4] elapsed = 294.27975mn
    Epoch[16580] : PPL = 1317.10 [# Gibbs steps=4] elapsed = 294.56630mn
    Epoch[16590] : PPL = 1509.37 [# Gibbs steps=4] elapsed = 294.85749mn
    Epoch[16600] : PPL = 1298.50 [# Gibbs steps=4] elapsed = 295.14762mn
    Epoch[16610] : PPL = 1256.47 [# Gibbs steps=4] elapsed = 295.43661mn
    Epoch[16620] : PPL = 1229.90 [# Gibbs steps=4] elapsed = 295.72712mn
    Epoch[16630] : PPL = 1230.94 [# Gibbs steps=4] elapsed = 296.01666mn
    Epoch[16640] : PPL = 1364.63 [# Gibbs steps=4] elapsed = 296.30938mn
    Epoch[16650] : PPL = 1919.93 [# Gibbs steps=4] elapsed = 296.59878mn
    Epoch[16660] : PPL = 1737.61 [# Gibbs steps=4] elapsed = 296.88497mn
    Epoch[16670] : PPL = 1587.87 [# Gibbs steps=4] elapsed = 297.17259mn
    Epoch[16680] : PPL = 871.49 [# Gibbs steps=4] elapsed = 297.45672mn
    Epoch[16690] : PPL = 897.88 [# Gibbs steps=4] elapsed = 297.74469mn
    Epoch[16700] : PPL = 1439.73 [# Gibbs steps=4] elapsed = 298.02973mn
    Epoch[16710] : PPL = 1236.84 [# Gibbs steps=4] elapsed = 298.31696mn
    Epoch[16720] : PPL = 1152.38 [# Gibbs steps=4] elapsed = 298.60536mn
    Epoch[16730] : PPL = 1436.38 [# Gibbs steps=4] elapsed = 298.89553mn
    Epoch[16740] : PPL = 1379.62 [# Gibbs steps=4] elapsed = 299.17699mn
    Epoch[16750] : PPL = 1675.66 [# Gibbs steps=4] elapsed = 299.46587mn
    Epoch[16760] : PPL = 1008.95 [# Gibbs steps=4] elapsed = 299.75221mn
    Epoch[16770] : PPL = 1364.60 [# Gibbs steps=4] elapsed = 300.04198mn
    Epoch[16780] : PPL = 1100.71 [# Gibbs steps=4] elapsed = 300.32706mn
    Epoch[16790] : PPL = 1116.96 [# Gibbs steps=4] elapsed = 300.61937mn
    Epoch[16800] : PPL = 1452.60 [# Gibbs steps=4] elapsed = 300.91188mn
    Epoch[16810] : PPL = 1467.60 [# Gibbs steps=4] elapsed = 301.20174mn
    Epoch[16820] : PPL = 1370.38 [# Gibbs steps=4] elapsed = 301.48640mn
    Epoch[16830] : PPL = 1440.35 [# Gibbs steps=4] elapsed = 301.77649mn
    Epoch[16840] : PPL = 1014.80 [# Gibbs steps=4] elapsed = 302.06817mn
    Epoch[16850] : PPL = 1419.87 [# Gibbs steps=4] elapsed = 302.35676mn
    Epoch[16860] : PPL = 1555.90 [# Gibbs steps=4] elapsed = 302.64132mn
    Epoch[16870] : PPL = 1274.20 [# Gibbs steps=4] elapsed = 302.93094mn
    Epoch[16880] : PPL = 1422.45 [# Gibbs steps=4] elapsed = 303.22052mn
    Epoch[16890] : PPL = 1580.22 [# Gibbs steps=4] elapsed = 303.50386mn
    Epoch[16900] : PPL = 1283.42 [# Gibbs steps=4] elapsed = 303.79083mn
    Epoch[16910] : PPL = 1300.47 [# Gibbs steps=4] elapsed = 304.08023mn
    Epoch[16920] : PPL = 1579.84 [# Gibbs steps=4] elapsed = 304.36746mn
    Epoch[16930] : PPL = 1447.69 [# Gibbs steps=4] elapsed = 304.65674mn
    Epoch[16940] : PPL = 1020.29 [# Gibbs steps=4] elapsed = 304.94165mn
    Epoch[16950] : PPL = 842.93 [# Gibbs steps=4] elapsed = 305.22898mn
    Epoch[16960] : PPL = 1557.67 [# Gibbs steps=4] elapsed = 305.51950mn
    Epoch[16970] : PPL = 1558.57 [# Gibbs steps=4] elapsed = 305.80866mn
    Epoch[16980] : PPL = 1430.59 [# Gibbs steps=4] elapsed = 306.09759mn
    Epoch[16990] : PPL = 1453.21 [# Gibbs steps=4] elapsed = 306.38585mn
    Epoch[17000] : PPL = 1400.02 [# Gibbs steps=4] elapsed = 306.85563mn
    Epoch[17010] : PPL = 1160.81 [# Gibbs steps=4] elapsed = 307.14889mn
    Epoch[17020] : PPL = 1321.04 [# Gibbs steps=4] elapsed = 307.43165mn
    Epoch[17030] : PPL = 1283.43 [# Gibbs steps=4] elapsed = 307.72422mn
    Epoch[17040] : PPL = 1592.70 [# Gibbs steps=4] elapsed = 308.01083mn
    Epoch[17050] : PPL = 1354.32 [# Gibbs steps=4] elapsed = 308.30017mn
    Epoch[17060] : PPL = 1278.22 [# Gibbs steps=4] elapsed = 308.59062mn
    Epoch[17070] : PPL = 880.63 [# Gibbs steps=4] elapsed = 308.88039mn
    Epoch[17080] : PPL = 1507.97 [# Gibbs steps=4] elapsed = 309.16858mn
    Epoch[17090] : PPL = 1150.60 [# Gibbs steps=4] elapsed = 309.45460mn
    Epoch[17100] : PPL = 1350.44 [# Gibbs steps=4] elapsed = 309.74252mn
    Epoch[17110] : PPL = 1160.33 [# Gibbs steps=4] elapsed = 310.03050mn
    Epoch[17120] : PPL = 1479.77 [# Gibbs steps=4] elapsed = 310.32031mn
    Epoch[17130] : PPL = 1564.47 [# Gibbs steps=4] elapsed = 310.60724mn
    Epoch[17140] : PPL = 945.71 [# Gibbs steps=4] elapsed = 310.89638mn
    Epoch[17150] : PPL = 1475.78 [# Gibbs steps=4] elapsed = 311.18662mn
    Epoch[17160] : PPL = 1010.10 [# Gibbs steps=4] elapsed = 311.47575mn
    Epoch[17170] : PPL = 1390.12 [# Gibbs steps=4] elapsed = 311.76185mn
    Epoch[17180] : PPL = 1352.49 [# Gibbs steps=4] elapsed = 312.05130mn
    Epoch[17190] : PPL = 2038.48 [# Gibbs steps=4] elapsed = 312.34312mn
    Epoch[17200] : PPL = 1158.80 [# Gibbs steps=4] elapsed = 312.63483mn
    Epoch[17210] : PPL = 871.99 [# Gibbs steps=4] elapsed = 312.92324mn
    Epoch[17220] : PPL = 1368.17 [# Gibbs steps=4] elapsed = 313.21094mn
    Epoch[17230] : PPL = 1423.38 [# Gibbs steps=4] elapsed = 313.50199mn
    Epoch[17240] : PPL = 1277.85 [# Gibbs steps=4] elapsed = 313.78812mn
    Epoch[17250] : PPL = 1045.19 [# Gibbs steps=4] elapsed = 314.07609mn
    Epoch[17260] : PPL = 1416.84 [# Gibbs steps=4] elapsed = 314.36712mn
    Epoch[17270] : PPL = 1315.90 [# Gibbs steps=4] elapsed = 314.65540mn
    Epoch[17280] : PPL = 938.04 [# Gibbs steps=4] elapsed = 314.94463mn
    Epoch[17290] : PPL = 1493.28 [# Gibbs steps=4] elapsed = 315.23243mn
    Epoch[17300] : PPL = 1514.88 [# Gibbs steps=4] elapsed = 315.52097mn
    Epoch[17310] : PPL = 1463.64 [# Gibbs steps=4] elapsed = 315.81317mn
    Epoch[17320] : PPL = 1501.43 [# Gibbs steps=4] elapsed = 316.10213mn
    Epoch[17330] : PPL = 1382.99 [# Gibbs steps=4] elapsed = 316.39259mn
    Epoch[17340] : PPL = 1317.26 [# Gibbs steps=4] elapsed = 316.68125mn
    Epoch[17350] : PPL = 939.34 [# Gibbs steps=4] elapsed = 316.96985mn
    Epoch[17360] : PPL = 1290.00 [# Gibbs steps=4] elapsed = 317.25819mn
    Epoch[17370] : PPL = 1956.06 [# Gibbs steps=4] elapsed = 317.54831mn
    Epoch[17380] : PPL = 1107.96 [# Gibbs steps=4] elapsed = 317.83404mn
    Epoch[17390] : PPL = 1461.63 [# Gibbs steps=4] elapsed = 318.12209mn
    Epoch[17400] : PPL = 1364.10 [# Gibbs steps=4] elapsed = 318.41105mn
    Epoch[17410] : PPL = 1066.00 [# Gibbs steps=4] elapsed = 318.69836mn
    Epoch[17420] : PPL = 1496.19 [# Gibbs steps=4] elapsed = 318.98914mn
    Epoch[17430] : PPL = 1295.63 [# Gibbs steps=4] elapsed = 319.27614mn
    Epoch[17440] : PPL = 1374.21 [# Gibbs steps=4] elapsed = 319.56188mn
    Epoch[17450] : PPL = 1907.25 [# Gibbs steps=4] elapsed = 319.84635mn
    Epoch[17460] : PPL = 1512.99 [# Gibbs steps=4] elapsed = 320.13793mn
    Epoch[17470] : PPL = 1497.41 [# Gibbs steps=4] elapsed = 320.42532mn
    Epoch[17480] : PPL = 1102.02 [# Gibbs steps=4] elapsed = 320.71164mn
    Epoch[17490] : PPL = 929.62 [# Gibbs steps=4] elapsed = 320.99889mn
    Epoch[17500] : PPL = 1226.53 [# Gibbs steps=4] elapsed = 321.46145mn
    Epoch[17510] : PPL = 929.99 [# Gibbs steps=4] elapsed = 321.74844mn
    Epoch[17520] : PPL = 1431.85 [# Gibbs steps=4] elapsed = 322.03862mn
    Epoch[17530] : PPL = 1335.63 [# Gibbs steps=4] elapsed = 322.32560mn
    Epoch[17540] : PPL = 1571.03 [# Gibbs steps=4] elapsed = 322.61454mn
    Epoch[17550] : PPL = 1064.45 [# Gibbs steps=4] elapsed = 322.90378mn
    Epoch[17560] : PPL = 1112.38 [# Gibbs steps=4] elapsed = 323.19263mn
    Epoch[17570] : PPL = 1296.59 [# Gibbs steps=4] elapsed = 323.48384mn
    Epoch[17580] : PPL = 1456.37 [# Gibbs steps=4] elapsed = 323.77404mn
    Epoch[17590] : PPL = 1383.86 [# Gibbs steps=4] elapsed = 324.06553mn
    Epoch[17600] : PPL = 1549.28 [# Gibbs steps=4] elapsed = 324.35262mn
    Epoch[17610] : PPL = 1590.15 [# Gibbs steps=4] elapsed = 324.64004mn
    Epoch[17620] : PPL = 1166.75 [# Gibbs steps=4] elapsed = 324.92888mn
    Epoch[17630] : PPL = 1259.37 [# Gibbs steps=4] elapsed = 325.21571mn
    Epoch[17640] : PPL = 1299.87 [# Gibbs steps=4] elapsed = 325.50203mn
    Epoch[17650] : PPL = 1577.95 [# Gibbs steps=4] elapsed = 325.78886mn
    Epoch[17660] : PPL = 1523.88 [# Gibbs steps=4] elapsed = 326.07709mn
    Epoch[17670] : PPL = 1534.41 [# Gibbs steps=4] elapsed = 326.36764mn
    Epoch[17680] : PPL = 1610.16 [# Gibbs steps=4] elapsed = 326.65358mn
    Epoch[17690] : PPL = 820.55 [# Gibbs steps=4] elapsed = 326.94360mn
    Epoch[17700] : PPL = 1480.60 [# Gibbs steps=4] elapsed = 327.23296mn
    Epoch[17710] : PPL = 1566.99 [# Gibbs steps=4] elapsed = 327.52250mn
    Epoch[17720] : PPL = 1162.97 [# Gibbs steps=4] elapsed = 327.80943mn
    Epoch[17730] : PPL = 1470.23 [# Gibbs steps=4] elapsed = 328.10062mn
    Epoch[17740] : PPL = 1325.21 [# Gibbs steps=4] elapsed = 328.39136mn
    Epoch[17750] : PPL = 1315.83 [# Gibbs steps=4] elapsed = 328.68101mn
    Epoch[17760] : PPL = 1535.45 [# Gibbs steps=4] elapsed = 328.96832mn
    Epoch[17770] : PPL = 1368.24 [# Gibbs steps=4] elapsed = 329.25746mn
    Epoch[17780] : PPL = 1043.63 [# Gibbs steps=4] elapsed = 329.54760mn
    Epoch[17790] : PPL = 1549.64 [# Gibbs steps=4] elapsed = 329.83375mn
    Epoch[17800] : PPL = 1124.65 [# Gibbs steps=4] elapsed = 330.12398mn
    Epoch[17810] : PPL = 1590.11 [# Gibbs steps=4] elapsed = 330.41329mn
    Epoch[17820] : PPL = 1436.48 [# Gibbs steps=4] elapsed = 330.70426mn
    Epoch[17830] : PPL = 1384.33 [# Gibbs steps=4] elapsed = 330.99086mn
    Epoch[17840] : PPL = 1447.86 [# Gibbs steps=4] elapsed = 331.27823mn
    Epoch[17850] : PPL = 1309.36 [# Gibbs steps=4] elapsed = 331.56806mn
    Epoch[17860] : PPL = 1374.42 [# Gibbs steps=4] elapsed = 331.85852mn
    Epoch[17870] : PPL = 1457.12 [# Gibbs steps=4] elapsed = 332.14786mn
    Epoch[17880] : PPL = 1031.09 [# Gibbs steps=4] elapsed = 332.43559mn
    Epoch[17890] : PPL = 1453.68 [# Gibbs steps=4] elapsed = 332.72708mn
    Epoch[17900] : PPL = 1073.46 [# Gibbs steps=4] elapsed = 333.01826mn
    Epoch[17910] : PPL = 1334.86 [# Gibbs steps=4] elapsed = 333.30900mn
    Epoch[17920] : PPL = 1639.76 [# Gibbs steps=4] elapsed = 333.59943mn
    Epoch[17930] : PPL = 1563.51 [# Gibbs steps=4] elapsed = 333.88625mn
    Epoch[17940] : PPL = 923.12 [# Gibbs steps=4] elapsed = 334.17066mn
    Epoch[17950] : PPL = 1407.57 [# Gibbs steps=4] elapsed = 334.46162mn
    Epoch[17960] : PPL = 1038.13 [# Gibbs steps=4] elapsed = 334.74829mn
    Epoch[17970] : PPL = 1249.20 [# Gibbs steps=4] elapsed = 335.03903mn
    Epoch[17980] : PPL = 1617.83 [# Gibbs steps=4] elapsed = 335.32763mn
    Epoch[17990] : PPL = 1034.12 [# Gibbs steps=4] elapsed = 335.61780mn
    Epoch[18000] : PPL = 921.18 [# Gibbs steps=4] elapsed = 336.08030mn
    Epoch[18010] : PPL = 1992.46 [# Gibbs steps=4] elapsed = 336.36866mn
    Epoch[18020] : PPL = 1336.07 [# Gibbs steps=4] elapsed = 336.66085mn
    Epoch[18030] : PPL = 984.71 [# Gibbs steps=4] elapsed = 336.94255mn
    Epoch[18040] : PPL = 1478.44 [# Gibbs steps=4] elapsed = 337.23303mn
    Epoch[18050] : PPL = 1306.03 [# Gibbs steps=4] elapsed = 337.52518mn
    Epoch[18060] : PPL = 1335.88 [# Gibbs steps=4] elapsed = 337.81552mn
    Epoch[18070] : PPL = 916.91 [# Gibbs steps=4] elapsed = 338.10743mn
    Epoch[18080] : PPL = 1033.97 [# Gibbs steps=4] elapsed = 338.39784mn
    Epoch[18090] : PPL = 1009.15 [# Gibbs steps=4] elapsed = 338.68559mn
    Epoch[18100] : PPL = 1293.12 [# Gibbs steps=4] elapsed = 338.96839mn
    Epoch[18110] : PPL = 1269.51 [# Gibbs steps=4] elapsed = 339.25377mn
    Epoch[18120] : PPL = 1034.02 [# Gibbs steps=4] elapsed = 339.54108mn
    Epoch[18130] : PPL = 1549.16 [# Gibbs steps=4] elapsed = 339.82858mn
    Epoch[18140] : PPL = 2057.81 [# Gibbs steps=4] elapsed = 340.11758mn
    Epoch[18150] : PPL = 1258.46 [# Gibbs steps=4] elapsed = 340.40861mn
    Epoch[18160] : PPL = 1356.38 [# Gibbs steps=4] elapsed = 340.69992mn
    Epoch[18170] : PPL = 1432.26 [# Gibbs steps=4] elapsed = 340.98910mn
    Epoch[18180] : PPL = 1561.33 [# Gibbs steps=4] elapsed = 341.27884mn
    Epoch[18190] : PPL = 1547.02 [# Gibbs steps=4] elapsed = 341.56664mn
    Epoch[18200] : PPL = 1958.03 [# Gibbs steps=4] elapsed = 341.85816mn
    Epoch[18210] : PPL = 1517.22 [# Gibbs steps=4] elapsed = 342.14272mn
    Epoch[18220] : PPL = 1311.31 [# Gibbs steps=4] elapsed = 342.42826mn
    Epoch[18230] : PPL = 1144.92 [# Gibbs steps=4] elapsed = 342.71386mn
    Epoch[18240] : PPL = 1437.97 [# Gibbs steps=4] elapsed = 343.00320mn
    Epoch[18250] : PPL = 901.79 [# Gibbs steps=4] elapsed = 343.29362mn
    Epoch[18260] : PPL = 1302.88 [# Gibbs steps=4] elapsed = 343.58431mn
    Epoch[18270] : PPL = 1312.71 [# Gibbs steps=4] elapsed = 343.87483mn
    Epoch[18280] : PPL = 1564.75 [# Gibbs steps=4] elapsed = 344.16325mn
    Epoch[18290] : PPL = 1383.11 [# Gibbs steps=4] elapsed = 344.44627mn
    Epoch[18300] : PPL = 1264.89 [# Gibbs steps=4] elapsed = 344.73299mn
    Epoch[18310] : PPL = 1324.31 [# Gibbs steps=4] elapsed = 345.02193mn
    Epoch[18320] : PPL = 1620.91 [# Gibbs steps=4] elapsed = 345.30871mn
    Epoch[18330] : PPL = 1544.30 [# Gibbs steps=4] elapsed = 345.59597mn
    Epoch[18340] : PPL = 1436.35 [# Gibbs steps=4] elapsed = 345.88205mn
    Epoch[18350] : PPL = 891.88 [# Gibbs steps=4] elapsed = 346.17127mn
    Epoch[18360] : PPL = 1315.22 [# Gibbs steps=4] elapsed = 346.46270mn
    Epoch[18370] : PPL = 1204.62 [# Gibbs steps=4] elapsed = 346.75112mn
    Epoch[18380] : PPL = 1301.56 [# Gibbs steps=4] elapsed = 347.04221mn
    Epoch[18390] : PPL = 1535.20 [# Gibbs steps=4] elapsed = 347.32990mn
    Epoch[18400] : PPL = 1151.20 [# Gibbs steps=4] elapsed = 347.61978mn
    Epoch[18410] : PPL = 1536.59 [# Gibbs steps=4] elapsed = 347.91083mn
    Epoch[18420] : PPL = 1229.14 [# Gibbs steps=4] elapsed = 348.19863mn
    Epoch[18430] : PPL = 1001.38 [# Gibbs steps=4] elapsed = 348.48535mn
    Epoch[18440] : PPL = 854.30 [# Gibbs steps=4] elapsed = 348.76767mn
    Epoch[18450] : PPL = 1470.25 [# Gibbs steps=4] elapsed = 349.05811mn
    Epoch[18460] : PPL = 1302.65 [# Gibbs steps=4] elapsed = 349.34832mn
    Epoch[18470] : PPL = 860.15 [# Gibbs steps=4] elapsed = 349.63428mn
    Epoch[18480] : PPL = 1263.91 [# Gibbs steps=4] elapsed = 349.92541mn
    Epoch[18490] : PPL = 1217.25 [# Gibbs steps=4] elapsed = 350.21658mn
    Epoch[18500] : PPL = 1360.96 [# Gibbs steps=4] elapsed = 350.68566mn
    Epoch[18510] : PPL = 1523.56 [# Gibbs steps=4] elapsed = 350.97450mn
    Epoch[18520] : PPL = 1457.98 [# Gibbs steps=4] elapsed = 351.26515mn
    Epoch[18530] : PPL = 1429.39 [# Gibbs steps=4] elapsed = 351.55393mn
    Epoch[18540] : PPL = 875.67 [# Gibbs steps=4] elapsed = 351.83957mn
    Epoch[18550] : PPL = 1992.54 [# Gibbs steps=4] elapsed = 352.12794mn
    Epoch[18560] : PPL = 1121.06 [# Gibbs steps=4] elapsed = 352.41736mn
    Epoch[18570] : PPL = 958.62 [# Gibbs steps=4] elapsed = 352.70466mn
    Epoch[18580] : PPL = 1511.90 [# Gibbs steps=4] elapsed = 352.99455mn
    Epoch[18590] : PPL = 1572.60 [# Gibbs steps=4] elapsed = 353.28196mn
    Epoch[18600] : PPL = 912.88 [# Gibbs steps=4] elapsed = 353.56537mn
    Epoch[18610] : PPL = 1406.83 [# Gibbs steps=4] elapsed = 353.85481mn
    Epoch[18620] : PPL = 1330.29 [# Gibbs steps=4] elapsed = 354.14843mn
    Epoch[18630] : PPL = 1369.33 [# Gibbs steps=4] elapsed = 354.43477mn
    Epoch[18640] : PPL = 1512.21 [# Gibbs steps=4] elapsed = 354.71969mn
    Epoch[18650] : PPL = 1437.94 [# Gibbs steps=4] elapsed = 355.00978mn
    Epoch[18660] : PPL = 1324.87 [# Gibbs steps=4] elapsed = 355.30083mn
    Epoch[18670] : PPL = 1667.12 [# Gibbs steps=4] elapsed = 355.59031mn
    Epoch[18680] : PPL = 1180.18 [# Gibbs steps=4] elapsed = 355.88043mn
    Epoch[18690] : PPL = 1224.34 [# Gibbs steps=4] elapsed = 356.16944mn
    Epoch[18700] : PPL = 1190.56 [# Gibbs steps=4] elapsed = 356.45732mn
    Epoch[18710] : PPL = 1131.22 [# Gibbs steps=4] elapsed = 356.74793mn
    Epoch[18720] : PPL = 869.74 [# Gibbs steps=4] elapsed = 357.03744mn
    Epoch[18730] : PPL = 1516.05 [# Gibbs steps=4] elapsed = 357.32409mn
    Epoch[18740] : PPL = 1810.35 [# Gibbs steps=4] elapsed = 357.61514mn
    Epoch[18750] : PPL = 1443.24 [# Gibbs steps=4] elapsed = 357.90566mn
    Epoch[18760] : PPL = 1481.65 [# Gibbs steps=4] elapsed = 358.19502mn
    Epoch[18770] : PPL = 1048.60 [# Gibbs steps=4] elapsed = 358.48365mn
    Epoch[18780] : PPL = 1538.43 [# Gibbs steps=4] elapsed = 358.77025mn
    Epoch[18790] : PPL = 1351.85 [# Gibbs steps=4] elapsed = 359.05248mn
    Epoch[18800] : PPL = 1290.47 [# Gibbs steps=4] elapsed = 359.34058mn
    Epoch[18810] : PPL = 1169.56 [# Gibbs steps=4] elapsed = 359.63275mn
    Epoch[18820] : PPL = 1041.18 [# Gibbs steps=4] elapsed = 359.91923mn
    Epoch[18830] : PPL = 1139.58 [# Gibbs steps=4] elapsed = 360.20524mn
    Epoch[18840] : PPL = 1388.51 [# Gibbs steps=4] elapsed = 360.49239mn
    Epoch[18850] : PPL = 1482.30 [# Gibbs steps=4] elapsed = 360.78057mn
    Epoch[18860] : PPL = 1595.80 [# Gibbs steps=4] elapsed = 361.06916mn
    Epoch[18870] : PPL = 1239.76 [# Gibbs steps=4] elapsed = 361.35205mn
    Epoch[18880] : PPL = 1294.51 [# Gibbs steps=4] elapsed = 361.64127mn
    Epoch[18890] : PPL = 1469.40 [# Gibbs steps=4] elapsed = 361.93099mn
    Epoch[18900] : PPL = 998.73 [# Gibbs steps=4] elapsed = 362.22056mn
    Epoch[18910] : PPL = 1101.84 [# Gibbs steps=4] elapsed = 362.51059mn
    Epoch[18920] : PPL = 1333.15 [# Gibbs steps=4] elapsed = 362.80099mn
    Epoch[18930] : PPL = 1273.69 [# Gibbs steps=4] elapsed = 363.08874mn
    Epoch[18940] : PPL = 1254.27 [# Gibbs steps=4] elapsed = 363.38058mn
    Epoch[18950] : PPL = 1943.77 [# Gibbs steps=4] elapsed = 363.67178mn
    Epoch[18960] : PPL = 1323.17 [# Gibbs steps=4] elapsed = 363.96350mn
    Epoch[18970] : PPL = 1506.49 [# Gibbs steps=4] elapsed = 364.25222mn
    Epoch[18980] : PPL = 908.24 [# Gibbs steps=4] elapsed = 364.54345mn
    Epoch[18990] : PPL = 1307.36 [# Gibbs steps=4] elapsed = 364.83044mn
    Epoch[19000] : PPL = 1480.55 [# Gibbs steps=4] elapsed = 365.29945mn
    Epoch[19010] : PPL = 1644.05 [# Gibbs steps=4] elapsed = 365.58852mn
    Epoch[19020] : PPL = 1225.51 [# Gibbs steps=4] elapsed = 365.87876mn
    Epoch[19030] : PPL = 914.10 [# Gibbs steps=4] elapsed = 366.16650mn
    Epoch[19040] : PPL = 1519.06 [# Gibbs steps=4] elapsed = 366.45512mn
    Epoch[19050] : PPL = 1056.46 [# Gibbs steps=4] elapsed = 366.74516mn
    Epoch[19060] : PPL = 1301.37 [# Gibbs steps=4] elapsed = 367.03814mn
    Epoch[19070] : PPL = 1432.17 [# Gibbs steps=4] elapsed = 367.32841mn
    Epoch[19080] : PPL = 1515.61 [# Gibbs steps=4] elapsed = 367.62090mn
    Epoch[19090] : PPL = 1611.93 [# Gibbs steps=4] elapsed = 367.90875mn
    Epoch[19100] : PPL = 1469.69 [# Gibbs steps=4] elapsed = 368.19922mn
    Epoch[19110] : PPL = 1555.19 [# Gibbs steps=4] elapsed = 368.48885mn
    Epoch[19120] : PPL = 1476.28 [# Gibbs steps=4] elapsed = 368.77789mn
    Epoch[19130] : PPL = 1367.66 [# Gibbs steps=4] elapsed = 369.06876mn
    Epoch[19140] : PPL = 1418.57 [# Gibbs steps=4] elapsed = 369.36175mn
    Epoch[19150] : PPL = 1550.51 [# Gibbs steps=4] elapsed = 369.65154mn
    Epoch[19160] : PPL = 1444.93 [# Gibbs steps=4] elapsed = 369.94271mn
    Epoch[19170] : PPL = 1546.61 [# Gibbs steps=4] elapsed = 370.23132mn
    Epoch[19180] : PPL = 1432.22 [# Gibbs steps=4] elapsed = 370.51859mn
    Epoch[19190] : PPL = 1319.65 [# Gibbs steps=4] elapsed = 370.80707mn
    Epoch[19200] : PPL = 1020.96 [# Gibbs steps=4] elapsed = 371.09131mn
    Epoch[19210] : PPL = 1693.84 [# Gibbs steps=4] elapsed = 371.38459mn
    Epoch[19220] : PPL = 1002.93 [# Gibbs steps=4] elapsed = 371.67319mn
    Epoch[19230] : PPL = 1496.16 [# Gibbs steps=4] elapsed = 371.96386mn
    Epoch[19240] : PPL = 1292.87 [# Gibbs steps=4] elapsed = 372.25233mn
    Epoch[19250] : PPL = 1567.78 [# Gibbs steps=4] elapsed = 372.54122mn
    Epoch[19260] : PPL = 1230.95 [# Gibbs steps=4] elapsed = 372.83317mn
    Epoch[19270] : PPL = 1391.98 [# Gibbs steps=4] elapsed = 373.12337mn
    Epoch[19280] : PPL = 1169.28 [# Gibbs steps=4] elapsed = 373.41364mn
    Epoch[19290] : PPL = 1290.22 [# Gibbs steps=4] elapsed = 373.70013mn
    Epoch[19300] : PPL = 1026.98 [# Gibbs steps=4] elapsed = 373.98792mn
    Epoch[19310] : PPL = 1149.29 [# Gibbs steps=4] elapsed = 374.27546mn
    Epoch[19320] : PPL = 1134.62 [# Gibbs steps=4] elapsed = 374.56011mn
    Epoch[19330] : PPL = 873.33 [# Gibbs steps=4] elapsed = 374.84853mn
    Epoch[19340] : PPL = 1344.60 [# Gibbs steps=4] elapsed = 375.13815mn
    Epoch[19350] : PPL = 1649.97 [# Gibbs steps=4] elapsed = 375.42841mn
    Epoch[19360] : PPL = 1367.51 [# Gibbs steps=4] elapsed = 375.71832mn
    Epoch[19370] : PPL = 1617.49 [# Gibbs steps=4] elapsed = 376.00756mn
    Epoch[19380] : PPL = 947.68 [# Gibbs steps=4] elapsed = 376.30156mn
    Epoch[19390] : PPL = 2003.90 [# Gibbs steps=4] elapsed = 376.59168mn
    Epoch[19400] : PPL = 1468.85 [# Gibbs steps=4] elapsed = 376.88335mn
    Epoch[19410] : PPL = 1432.32 [# Gibbs steps=4] elapsed = 377.17263mn
    Epoch[19420] : PPL = 1378.36 [# Gibbs steps=4] elapsed = 377.46001mn
    Epoch[19430] : PPL = 1116.76 [# Gibbs steps=4] elapsed = 377.74495mn
    Epoch[19440] : PPL = 1316.33 [# Gibbs steps=4] elapsed = 378.02866mn
    Epoch[19450] : PPL = 1145.79 [# Gibbs steps=4] elapsed = 378.31733mn
    Epoch[19460] : PPL = 1267.10 [# Gibbs steps=4] elapsed = 378.60334mn
    Epoch[19470] : PPL = 1457.96 [# Gibbs steps=4] elapsed = 378.89440mn
    Epoch[19480] : PPL = 1293.72 [# Gibbs steps=4] elapsed = 379.18284mn
    Epoch[19490] : PPL = 1514.73 [# Gibbs steps=4] elapsed = 379.47094mn
    Epoch[19500] : PPL = 1507.61 [# Gibbs steps=4] elapsed = 379.94051mn
    Epoch[19510] : PPL = 1496.62 [# Gibbs steps=4] elapsed = 380.22870mn
    Epoch[19520] : PPL = 1114.47 [# Gibbs steps=4] elapsed = 380.51758mn
    Epoch[19530] : PPL = 1665.80 [# Gibbs steps=4] elapsed = 380.81119mn
    Epoch[19540] : PPL = 922.49 [# Gibbs steps=4] elapsed = 381.09904mn
    Epoch[19550] : PPL = 1517.27 [# Gibbs steps=4] elapsed = 381.39057mn
    Epoch[19560] : PPL = 1552.84 [# Gibbs steps=4] elapsed = 381.68081mn
    Epoch[19570] : PPL = 1381.26 [# Gibbs steps=4] elapsed = 381.96627mn
    Epoch[19580] : PPL = 1409.11 [# Gibbs steps=4] elapsed = 382.25499mn
    Epoch[19590] : PPL = 1354.47 [# Gibbs steps=4] elapsed = 382.54633mn
    Epoch[19600] : PPL = 1616.13 [# Gibbs steps=4] elapsed = 382.83656mn
    Epoch[19610] : PPL = 1355.50 [# Gibbs steps=4] elapsed = 383.12398mn
    Epoch[19620] : PPL = 1098.47 [# Gibbs steps=4] elapsed = 383.41204mn
    Epoch[19630] : PPL = 1529.80 [# Gibbs steps=4] elapsed = 383.70313mn
    Epoch[19640] : PPL = 1404.09 [# Gibbs steps=4] elapsed = 383.99441mn
    Epoch[19650] : PPL = 1516.57 [# Gibbs steps=4] elapsed = 384.28331mn
    Epoch[19660] : PPL = 2032.79 [# Gibbs steps=4] elapsed = 384.57022mn
    Epoch[19670] : PPL = 1539.05 [# Gibbs steps=4] elapsed = 384.86152mn
    Epoch[19680] : PPL = 1382.22 [# Gibbs steps=4] elapsed = 385.15186mn
    Epoch[19690] : PPL = 1733.76 [# Gibbs steps=4] elapsed = 385.44379mn
    Epoch[19700] : PPL = 1556.67 [# Gibbs steps=4] elapsed = 385.73440mn
    Epoch[19710] : PPL = 1506.29 [# Gibbs steps=4] elapsed = 386.02341mn
    Epoch[19720] : PPL = 1456.94 [# Gibbs steps=4] elapsed = 386.31489mn
    Epoch[19730] : PPL = 928.37 [# Gibbs steps=4] elapsed = 386.60362mn
    Epoch[19740] : PPL = 926.44 [# Gibbs steps=4] elapsed = 386.89337mn
    Epoch[19750] : PPL = 1505.63 [# Gibbs steps=4] elapsed = 387.18730mn
    Epoch[19760] : PPL = 1564.39 [# Gibbs steps=4] elapsed = 387.47773mn
    Epoch[19770] : PPL = 1616.84 [# Gibbs steps=4] elapsed = 387.76644mn
    Epoch[19780] : PPL = 1115.43 [# Gibbs steps=4] elapsed = 388.05556mn
    Epoch[19790] : PPL = 1433.22 [# Gibbs steps=4] elapsed = 388.34412mn
    Epoch[19800] : PPL = 1309.16 [# Gibbs steps=4] elapsed = 388.63312mn
    Epoch[19810] : PPL = 1287.88 [# Gibbs steps=4] elapsed = 388.92292mn
    Epoch[19820] : PPL = 1443.72 [# Gibbs steps=4] elapsed = 389.20684mn
    Epoch[19830] : PPL = 1294.99 [# Gibbs steps=4] elapsed = 389.49186mn
    Epoch[19840] : PPL = 1607.93 [# Gibbs steps=4] elapsed = 389.77947mn
    Epoch[19850] : PPL = 1015.38 [# Gibbs steps=4] elapsed = 390.06687mn
    Epoch[19860] : PPL = 1442.10 [# Gibbs steps=4] elapsed = 390.35963mn
    Epoch[19870] : PPL = 1442.19 [# Gibbs steps=4] elapsed = 390.64713mn
    Epoch[19880] : PPL = 1556.42 [# Gibbs steps=4] elapsed = 390.93515mn
    Epoch[19890] : PPL = 1383.73 [# Gibbs steps=4] elapsed = 391.22061mn
    Epoch[19900] : PPL = 1459.45 [# Gibbs steps=4] elapsed = 391.51469mn
    Epoch[19910] : PPL = 1434.59 [# Gibbs steps=4] elapsed = 391.80614mn
    Epoch[19920] : PPL = 1182.40 [# Gibbs steps=4] elapsed = 392.09688mn
    Epoch[19930] : PPL = 1387.69 [# Gibbs steps=4] elapsed = 392.38700mn
    Epoch[19940] : PPL = 892.07 [# Gibbs steps=4] elapsed = 392.67524mn
    Epoch[19950] : PPL = 1413.96 [# Gibbs steps=4] elapsed = 392.96330mn
    Epoch[19960] : PPL = 1146.92 [# Gibbs steps=4] elapsed = 393.25282mn
    Epoch[19970] : PPL = 1336.73 [# Gibbs steps=4] elapsed = 393.54430mn
    Epoch[19980] : PPL = 1341.65 [# Gibbs steps=4] elapsed = 393.83276mn
    Epoch[19990] : PPL = 880.55 [# Gibbs steps=4] elapsed = 394.12114mn


maybe this could work to **visualize** the categories ?


    cat_list = ["restaurants", "afghani", "african", "senegalese",
"southafrican", "New", "newamerican", "Traditional", "tradamerican", "arabian",
"argentine", "armenian", "asianfusion", "australian", "austrian", "bangladeshi",
"bbq", "basque", "belgian", "brasseries", "brazilian", "breakfast_brunch",
"british", "buffets", "burgers", "burmese", "cafes", "cafeteria", "cajun",
"cambodian", "caribbean", "dominican", "haitian", "puertorican", "trinidadian",
"catalan", "cheesesteaks", "chicken_wings", "chinese", "cantonese", "dimsum",
"shanghainese", "szechuan", "comfortfood", "creperies", "cuban", "czech",
"delis", "diners", "ethiopian", "hotdogs", "filipino", "fishnchips", "fondue",
"food_court", "foodstands", "french", "gastropubs", "german", "gluten_free",
"greek", "halal", "hawaiian", "himalayan", "hotdog", "hotpot", "hungarian",
"iberian", "indpak", "indonesian", "irish", "italian", "japanese", "korean",
"kosher", "laotian", "latin", "colombian", "salvadoran", "venezuelan",
"raw_food", "malaysian", "mediterranean", "falafel", "mexican", "mideastern",
"egyptian", "lebanese", "modern_european", "mongolian", "moroccan", "pakistani",
"persian", "peruvian", "pizza", "polish", "portuguese", "russian", "salad",
"sandwiches", "scandinavian", "scottish", "seafood", "singaporean", "slovakian",
"soulfood", "soup", "southern", "spanish", "steak", "sushi", "taiwanese",
"tapas", "tapasmallplates", "tex-mex", "thai", "turkish", "ukrainian", "vegan",
"vegetarian", "vietnamese"]
    #cat_list = ["$", "$$", "$$$"]
    (label_cat_lexicon, reverse_label_cat_lexicon) =
utils.create_lexicon_from_strings(cat_list)
    rl_cat = utils.ResourceLabeler(lexicon = label_cat_lexicon,
accessor="categories")
    categorized = rl_cat.process(some_batch)
    extended_cat_lexicon = reverse_label_cat_lexicon + ['unknown']
    reverse_point_categories = {}
    for index, category in enumerate(categorized):
        if reverse_point_categories.get(category) != None:
            reverse_point_categories[category] =
np.append(reverse_point_categories[category], X_2d[index, :].reshape(1, 2),
axis=0)
        else:
            reverse_point_categories[category] = np.zeros([1,2])
            reverse_point_categories[category][0, :] = X_2d[index, :]

    for key in reverse_point_categories.keys():
        plt.scatter(reverse_point_categories[key][:, 0],
reverse_point_categories[key][:, 1], cmap=mpl.cm.summer,
c=key*np.ones(len(reverse_point_categories[key])),
label=extended_cat_lexicon[key]);
    plt.legend(scatterpoints=1)


    # create a labeling lexicon:
    (label_lexicon, reverse_label_lexicon) = utils.create_lexicon_from_strings(["restaurants", "afghani", "african", "senegalese", "southafrican", "New", "newamerican", "Traditional", "tradamerican", "arabian", "argentine", "armenian", "asianfusion", "australian", "austrian", "bangladeshi", "bbq", "basque", "belgian", "brasseries", "brazilian", "breakfast_brunch", "british", "buffets", "burgers", "burmese", "cafes", "cafeteria", "cajun", "cambodian", "caribbean", "dominican", "haitian", "puertorican", "trinidadian", "catalan", "cheesesteaks", "chicken_wings", "chinese", "cantonese", "dimsum", "shanghainese", "szechuan", "comfortfood", "creperies", "cuban", "czech", "delis", "diners", "ethiopian", "hotdogs", "filipino", "fishnchips", "fondue", "food_court", "foodstands", "french", "gastropubs", "german", "gluten_free", "greek", "halal", "hawaiian", "himalayan", "hotdog", "hotpot", "hungarian", "iberian", "indpak", "indonesian", "irish", "italian", "japanese", "korean", "kosher", "laotian", "latin", "colombian", "salvadoran", "venezuelan", "raw_food", "malaysian", "mediterranean", "falafel", "mexican", "mideastern", "egyptian", "lebanese", "modern_european", "mongolian", "moroccan", "pakistani", "persian", "peruvian", "pizza", "polish", "portuguese", "russian", "salad", "sandwiches", "scandinavian", "scottish", "seafood", "singaporean", "slovakian", "soulfood", "soup", "southern", "spanish", "steak", "sushi", "taiwanese", "tapas", "tapasmallplates", "tex-mex", "thai", "turkish", "ukrainian", "vegan", "vegetarian", "vietnamese"])


    trial_batch = Batch(
        data=utils.mongo_database_global['restaurants'].find(), # from Mongo's cursor enumerator
        batch_size = 2000,  # mini-batch
        shuffle = True, # stochastic
        conversion = rc.process # convert to matrices using lexicon)
    )
    rl = utils.ResourceLabeler(lexicon = label_lexicon, accessor="categories")
    some_batch = trial_batch.get_raw_batch()
    (hidden, scaling) = encoder.project_into_hidden_layer(batch.conversion(some_batch))
    X_2d = bh_sne(hidden.astype('float64'))


    plt.scatter(X_2d[:,0], X_2d[:, 1], c =rl.process(some_batch) )




    <matplotlib.collections.PathCollection at 0x120010650>




![png](Mongo%20Stochastic%20Gradient%20Descent%20T-SNE_files/Mongo%20Stochastic%20Gradient%20Descent%20T-SNE_16_1.png)



    from mpl_toolkits.mplot3d.axes3d import Axes3D
    datum = bh_sne(hidden.astype('float64'), d=3)
    fig = plt.figure()
    axis = Axes3D(fig = fig)
    axis.scatter(datum[:, 0], datum[:, 1], datum[:, 2], c=rl.process(some_batch))




    <mpl_toolkits.mplot3d.art3d.Path3DCollection at 0x120d96410>




![png](Mongo%20Stochastic%20Gradient%20Descent%20T-SNE_files/Mongo%20Stochastic%20Gradient%20Descent%20T-SNE_17_1.png)



    (label_lexicon, reverse_label_lexicon) = utils.create_lexicon_from_strings(["$", "$$", "$$$", "$$$$"])
    rl = utils.ResourceLabeler(lexicon = label_lexicon, accessor="price")
    (hidden, scaling) = encoder.project_into_hidden_layer(batch.conversion(some_batch))
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    datum = bh_sne(hidden.astype('float64'), d=3)
    fig = plt.figure()
    axis = Axes3D(fig = fig, rect= [0.1, 0.1, 2.0, 2.0])
    axis.scatter(datum[:, 0], datum[:, 1], datum[:, 2], c=rl.process(some_batch))
    axis.legend(reverse_label_lexicon + ['unknown'])




    <matplotlib.legend.Legend at 0x11738b2d0>




![png](Mongo%20Stochastic%20Gradient%20Descent%20T-SNE_files/Mongo%20Stochastic%20Gradient%20Descent%20T-SNE_18_1.png)



    (rating_lexicon, reverse_rating_lexicon) = utils.create_lexicon_from_strings(["0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0"])
    rl = utils.ResourceLabeler(lexicon = rating_lexicon, accessor="rating", converter = lambda i: ("%.1f" % (round(i * 2) / 2.0)))
    (hidden, scaling) = encoder.project_into_hidden_layer(batch.conversion(some_batch))
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    datum = bh_sne(hidden.astype('float64'), d=3)
    fig = plt.figure()
    axis = Axes3D(fig = fig, rect= [0.1, 0.1, 2.0, 2.0])
    axis.scatter(datum[:, 0], datum[:, 1], datum[:, 2], c=rl.process(some_batch), cmap=mpl.cm.binary)
    axis.legend(reverse_label_lexicon + ['unknown'])




    <matplotlib.legend.Legend at 0x1195f5350>




![png](Mongo%20Stochastic%20Gradient%20Descent%20T-SNE_files/Mongo%20Stochastic%20Gradient%20Descent%20T-SNE_19_1.png)



    fig = plt.figure()
    axis = Axes3D(fig = fig, rect= [0.1, 0.1, 2.0, 2.0])
    axis.scatter(datum[:, 0], datum[:, 1], datum[:, 2], c=rl.process(some_batch), cmap=mpl.cm.binary)
    axis.legend(reverse_label_lexicon + ['unknown'])




    <matplotlib.legend.Legend at 0x118e188d0>




![png](Mongo%20Stochastic%20Gradient%20Descent%20T-SNE_files/Mongo%20Stochastic%20Gradient%20Descent%20T-SNE_20_1.png)



    z = np.zeros([1,2])
    z[0,:] = [1, 2]
    np.append(z, np.array([[1, 2]]), axis=0)




    array([[ 1.,  2.],
           [ 1.,  2.]])




    encoder.save("backup.pkz")


    


![png](Mongo%20Stochastic%20Gradient%20Descent%20T-SNE_files/Mongo%20Stochastic%20Gradient%20Descent%20T-SNE_23_0.png)



    
