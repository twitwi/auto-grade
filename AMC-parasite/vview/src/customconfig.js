
let cfg = {
  pyConnection: `//${window.document.domain}:5000`,
  defaultProjectDir: '2018-infospichi-3-exam-2',
  classes: '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt' + "=:;.,-_()[]!?*/'"
}

console.log(`HELP: use\n   localStorage.setItem('cfg--defaultProjectDir', '${cfg.defaultProjectDir}')`)

for (let k in cfg) {
  let lk = 'cfg--' + k
  let v = localStorage.getItem(lk)
  if (v != null) {
    cfg[k] = v
  }
}

export default cfg
