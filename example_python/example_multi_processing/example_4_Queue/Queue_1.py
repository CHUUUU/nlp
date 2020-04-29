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
        print(j)


# 특징 -=> main process 와 subprocess 가 각각 실행 (생각보다 subprocess 가 느리게 실행)

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


# ...
# ...


# 1386
# 1387
# 1388
# 1389
# 1390
# 1391
# 1392
# 1393
# 1394
# 1395
# 1396
# come on baby
# 1397
# come on baby
# 1398
# come on baby
# 1399
# come on baby
# 1400
# come on baby
# 1401
# come on baby
# 1402
# come on baby
# 1403
# come on baby
# 1404
# come on baby
# 1405
# come on baby
# 1406
# come on baby
# 1407
# come on baby
# 1408
# come on baby
# 1409
# come on baby
# 1410
# come on baby
# 1411
# come on baby
# 1412
# come on baby
# 1413
# come on baby
# 1414
# come on baby
# 1415
# come on baby
# 1416
# come on baby
# 1417
# come on baby
# 1418
# come on baby
# 1419
# come on baby
# 1420
# come on baby
# 1421
# come on baby
# 1422
# come on baby
# 1423
# come on baby
# 1424
# come on baby
# 1425
# come on baby
# 1426
# come on baby
# 1427
# come on baby
# 1428
# come on baby
# 1429
# come on baby
# 1430
# come on baby
# 1431
# come on baby
# 1432
# come on baby
# 1433
# come on baby
# 1434
# come on baby
# 1435
# come on baby
# 1436
# come on baby
# 1437
# come on baby
# 1438
# come on baby
# 1439
# come on baby
# 1440
# come on baby
# 1441
# come on baby
# 1442
# come on baby
# 1443
# come on baby
# 1444
# come on baby
# 1445
# come on baby
# 1446
# come on baby
# 1447
# come on baby
# 1448
# come on baby
# 1449
# come on baby
# 1450
# come on baby
# 1451
# come on baby
# 1452
# come on baby
# 1453
# come on baby
# 1454
# come on baby
# 1455
# come on baby
# 1456
# come on baby
# 1457
# come on baby
# 1458
# come on baby
# 1459
# come on baby
# 1460
# come on baby
# 1461
# come on baby
# 1462
# come on baby
# 1463
# come on baby
# 1464
# come on baby
# 1465
# come on baby
# 1466
# come on baby
# 1467
# come on baby
# 1468
# come on baby
# 1469
# come on baby
# 1470
# come on baby
# 1471
# come on baby
# 1472
# come on baby
# 1473
# come on baby
# 1474
# come on baby
# 1475
# come on baby
# 1476
# come on baby
# 1477
# come on baby
# 1478
# come on baby
# 1479
# come on baby
# 1480
# come on baby
# 1481
# come on baby
# 1482
# come on baby
# 1483
# come on baby
# 1484
# come on baby
# 1485
# come on baby
# 1486
# come on baby
# 1487
# come on baby
# 1488
# come on baby
# 1489
# come on baby
# 1490
# come on baby
# 1491
# come on baby
# 1492
# come on baby
# 1493
# come on baby
# 1494
# come on baby
# 1495
# come on baby
# 1496
# 1497
# 1498
# 1499
# 1500
# 1501
# 1502
# 1503


# ...
# ...


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


