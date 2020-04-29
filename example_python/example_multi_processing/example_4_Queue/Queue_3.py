from multiprocessing import Process, Queue

def f(q):
    for i in range(0,100):
        print("come on baby")
    q.put([42, None, 'hello'])


if __name__ == '__main__':
    q = Queue()
    p = Process(target=f, args=(q,))
    p.start()
    for j in range(0, 2000):
        if j == 1800:
            print(q.get())
        print(j)


# 특징 main process 와 subprocess 가 각각 실행되다가 1800 에서 subprocess 가 실행될때까지 기다려줌

# 0
# 1
# 2
# 3
# 4
# 5
# 6
# 7
# 8
# 9
# 10
# 11
# 12
# 13
# 14
# 15
# 16
# 17


# ...
# ...


# 1276
# 1277
# 1278
# 1279
# 1280
# 1281
# 1282
# 1283
# 1284
# 1285
# 1286
# 1287
# 1288
# 1289
# 1290
# 1291
# 1292
# 1293
# 1294
# 1295
# come on baby
# 1296
# come on baby
# 1297
# come on baby
# 1298
# come on baby
# 1299
# come on baby
# 1300
# come on baby
# 1301
# come on baby
# 1302
# come on baby
# 1303
# 1304
# come on baby
# 1305
# come on baby
# 1306
# come on baby
# 1307
# come on baby
# 1308
# come on baby
# 1309
# come on baby
# 1310
# come on baby
# 1311
# come on baby
# 1312
# come on baby
# 1313
# come on baby
# 1314
# come on baby
# 1315
# come on baby
# 1316
# come on baby
# 1317
# come on baby
# 1318
# come on baby
# 1319
# come on baby
# 1320
# come on baby
# 1321
# come on baby
# 1322
# come on baby
# 1323
# come on baby
# 1324
# come on baby
# 1325
# come on baby
# 1326
# come on baby
# 1327
# come on baby
# 1328
# come on baby
# 1329
# come on baby
# 1330
# come on baby
# 1331
# come on baby
# 1332
# come on baby
# 1333
# come on baby
# 1334
# come on baby
# 1335
# come on baby
# 1336
# come on baby
# 1337
# come on baby
# 1338
# come on baby
# 1339
# come on baby
# 1340
# come on baby
# 1341
# come on baby
# 1342
# come on baby
# 1343
# come on baby
# 1344
# come on baby
# 1345
# come on baby
# 1346
# come on baby
# 1347
# come on baby
# 1348
# come on baby
# 1349
# come on baby
# 1350
# come on baby
# 1351
# come on baby
# 1352
# come on baby
# 1353
# come on baby
# 1354
# come on baby
# 1355
# come on baby
# 1356
# come on baby
# 1357
# come on baby
# 1358
# come on baby
# 1359
# come on baby
# 1360
# come on baby
# 1361
# come on baby
# 1362
# come on baby
# 1363
# come on baby
# 1364
# come on baby
# 1365
# come on baby
# 1366
# come on baby
# 1367
# come on baby
# 1368
# come on baby
# 1369
# come on baby
# 1370
# come on baby
# 1371
# come on baby
# 1372
# come on baby
# 1373
# come on baby
# 1374
# come on baby
# 1375
# come on baby
# 1376
# come on baby
# 1377
# come on baby
# 1378
# come on baby
# 1379
# come on baby
# 1380
# come on baby
# 1381
# come on baby
# 1382
# come on baby
# 1383
# come on baby
# 1384
# come on baby
# 1385
# come on baby
# 1386
# come on baby
# 1387
# come on baby
# 1388
# come on baby
# 1389
# come on baby
# 1390
# come on baby
# 1391
# come on baby
# 1392
# come on baby
# 1393
# come on baby
# 1394
# come on baby
# 1395
# come on baby
# 1396
# 1397
# 1398
# 1399
# 1400
# 1401
# 1402
# 1403
# 1404
# 1405


# ...
# ...


# 1786
# 1787
# 1788
# 1789
# 1790
# 1791
# 1792
# 1793
# 1794
# 1795
# 1796
# 1797
# 1798
# 1799
# [42, None, 'hello']
# 1800
# 1801
# 1802
# 1803
# 1804
# 1805
# 1806
# 1807
# 1808
# 1809


# ...
# ...


# 1989
# 1990
# 1991
# 1992
# 1993
# 1994
# 1995
# 1996
# 1997
# 1998
# 1999
