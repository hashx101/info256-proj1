def genPrimes():
    def isPrime(n):
        if n % 2 == 0: return n == 2
        d = 3
        while d * d <= n:
            if n % d == 0: return False
            d += 2
        return True
    def init(): # change to Sieve of Eratosthenes
        ps, qs, sieve = [], [], [True] * 50000
        p, m = 3, 0
        while p * p <= 100000:
            if isPrime(p):
                ps.insert(0, p)
                qs.insert(0, p + (p-1) / 2)
                m += 1
            p += 2
        for i in xrange(m):
            for j in xrange(qs[i], 50000, ps[i]):
                sieve[j] = False
        return m, ps, qs, sieve
    def advance(m, ps, qs, sieve, bottom):
        for i in xrange(50000): sieve[i] = True
        for i in xrange(m):
            qs[i] = (qs[i] - 50000) % ps[i]
        p = ps[0] + 2
        while p * p <= bottom + 100000:
            if isPrime(p):
                ps.insert(0, p)
                qs.insert(0, (p*p - bottom - 1)/2)
                m += 1
            p += 2
        for i in xrange(m):
            for j in xrange(qs[i], 50000, ps[i]):
                sieve[j] = False
        return m, ps, qs, sieve
    m, ps, qs, sieve = init()
    bottom, i = 0, 1
    yield 2
    while True:
        if i == 50000:
            bottom = bottom + 100000
            m, ps, qs, sieve = advance(m, ps, qs, sieve, bottom)
            i = 0
        elif sieve[i]:
            yield bottom + i + i + 1
            i += 1
        else: i += 1