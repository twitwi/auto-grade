<template>
  <div class="manual-label">
    <h1>{{ connected }}</h1>
    <h1>{{ error }}</h1>
    <h1>{{ message }}</h1>
    <input v-model="projectDir"/><br/>
    <button @click="loadLogs()">load logs</button><br/>
    <div v-for="(l,i) in logs" :key="i">
      <label>
        <input type="checkbox" v-model="l.selected">
        <span class="student">{{l.d[0]}}</span><br/>
        <span class="details">{{l}}</span>
      </label>
    </div>
    <button @click="exportMiniset()">export miniset</button><br/>
  </div>
</template>

<script>
import config from '../customconfig'
import { mapState } from 'vuex'
// mapGetters, mapMutations, mapActions

export default {
  name: 'MinisetBuilder',
  data () {
    return {
      projectDir: config.defaultProjectDir,
      logs: []
    }
  },
  sockets: {
    'miniset-got-logs': function (data) {
      this.logs = data.map(l => ({ d: JSON.parse(l), selected: false }))
    }
  },
  computed: {
    ...mapState(['connected', 'error', 'message'])
  },
  methods: {
    loadLogs () {
      this.$socket.emit('miniset-get-logs', {})
    },
    exportMiniset () {
      let name = new Date().toISOString().replace(/T.*/, '-' + Date.now())
      let pro = this.projectDir
      this.$socket.emit('miniset-export', { name, pro, annotations: this.logs.filter(l => l.selected).map(l => l.d) })
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.student { font-size: 150%; margin-right: 1em; }
.details { font-family: monospace; font-size: 70%; display: inline-block; overflow-x: scroll; }
</style>
