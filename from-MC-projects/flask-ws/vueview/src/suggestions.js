
function _all () {
  return {noms, prenoms, numbers, cost}
}

let m = new Map()
m.set('4A', 0.2)
m.set('3B', 0.2)
m.set('3g', 0.5)
m.set('3a', 0.9)
m.set('0O', 0.1)
m.set('0D', 0.11)
m.set('DO', 0.12)
m.set('1I', 0.1)
m.set('1i', 0.1)
m.set('1l', 0.1)
m.set('lI', 0.2)
m.set('zZ', 0.11)
m.set('2Z', 0.2)
m.set('2z', 0.2)

function cost (gt, pred) {
  if (gt === pred) return 0
  let res = m.get(gt + pred)
  if (res === undefined) {
    res = m.get(pred + gt)
  }
  if (res === undefined) {
    res = 1
  }
  // console.log('cost', gt, pred, res)
  return res
}

var numbers = 'test/toto/'.split('/')
var prenoms = 'test/toto'.split('/')
var noms = 'test/toto'.split('/')


export default _all()
