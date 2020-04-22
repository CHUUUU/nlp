class Cli(object):
  def __init__ ( self, client_dict ):
    s2c = multiprocessing.Queue()
    c2s = multiprocessing.Queue()
    p = multiprocessing.Process(target=self._subprocess_run, args=(client_dict,s2c,c2s))
    p.daemon = True
    p.start()
    
일반적으로 데몬쓰레드라고 하면, 메인이 죽으면 같이 죽는 쓰레드를 말한다.

데몬 쓰레드란 백그라운드에서 실행되는 쓰레드로 메인 쓰레드가 종료되면 즉시 종료되는 쓰레드이다. 
디폴트는 넌데몬이며, 해당 서브쓰레드는 메인 쓰레드가 종료할 지라도 자신의 작업이 끝날 때까지 계속 실행된다.
