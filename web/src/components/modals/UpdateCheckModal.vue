<script setup lang="ts">
import { ref, watch } from 'vue'
import { ExternalLink, RefreshCw } from 'lucide-vue-next'
import Modal from '@/components/ui/Modal.vue'
import Button from '@/components/ui/Button.vue'
import { checkUpdate, type UpdateStatus } from '@/api/models.ts'

const open = defineModel<boolean>({ required: true })
const loading = ref(false)
const error = ref('')
const result = ref<UpdateStatus | null>(null)
const statusLabels: Record<string, string> = {
  up_to_date: '当前已是主仓库最新版本', update_available: '主仓库有新更新',
  unknown: '当前版本信息不足，无法判断', local_ahead: '当前版本领先于主仓库', diverged: '当前版本与主仓库历史不同',
}
async function check() {
  loading.value = true
  error.value = ''
  result.value = null
  try { result.value = await checkUpdate() }
  catch (e) { error.value = e instanceof Error ? e.message : '更新检查失败' }
  finally { loading.value = false }
}
watch(open, value => { if (value) check() })
</script>

<template>
  <Modal
    v-model="open"
    title="检查主仓库更新"
    max-width="lg"
  >
    <div class="space-y-4 text-sm">
      <p
        v-if="loading"
        role="status"
        class="text-gray-500 flex items-center gap-2"
      >
        <RefreshCw class="w-4 h-4 animate-spin" />正在检查 GitHub 主仓库…
      </p>
      <p
        v-if="error"
        role="alert"
        class="rounded-xl p-4 bg-rose-50 text-rose-700"
      >
        {{ error }}
      </p>
      <template v-if="result">
        <p
          class="rounded-xl p-4 font-medium"
          :class="result.status === 'up_to_date' ? 'bg-emerald-50 text-emerald-800' : 'bg-brand-50 text-brand-800'"
        >
          {{ statusLabels[result.status] ?? result.status }}<span v-if="result.status === 'update_available' && result.ahead_by"> · {{ result.ahead_by }} 个新提交</span>
        </p>
        <dl class="grid grid-cols-[6rem_1fr] gap-x-3 gap-y-3 text-gray-600">
          <dt class="text-gray-400">
            当前版本
          </dt><dd class="font-mono break-all">
            {{ result.current_revision.slice(0, 12) }}
          </dd>
          <dt class="text-gray-400">
            主仓库最新
          </dt><dd class="font-mono break-all">
            {{ result.latest_revision.slice(0, 12) }} · {{ result.branch }}
          </dd>
          <dt
            v-if="result.custom_build"
            class="text-gray-400"
          >
            定制构建
          </dt><dd
            v-if="result.custom_build"
            class="break-all"
          >
            {{ result.custom_build }}
          </dd>
          <dt class="text-gray-400">
            提交时间
          </dt><dd>{{ new Date(result.committed_at).toLocaleString() }}</dd>
          <dt class="text-gray-400">
            检查时间
          </dt><dd>{{ new Date(result.checked_at * 1000).toLocaleString() }}</dd>
        </dl>
        <p class="whitespace-pre-wrap break-words text-xs leading-6 rounded-xl bg-gray-50 p-4 text-gray-600">
          {{ result.commit_message }}
        </p>
        <div class="flex gap-4 text-brand-600 text-xs">
          <a
            :href="result.commit_url"
            target="_blank"
            rel="noopener noreferrer"
            class="inline-flex items-center gap-1 hover:underline"
          >查看最新提交<ExternalLink class="w-3 h-3" /></a>
          <a
            v-if="result.compare_url"
            :href="result.compare_url"
            target="_blank"
            rel="noopener noreferrer"
            class="inline-flex items-center gap-1 hover:underline"
          >查看更新差异<ExternalLink class="w-3 h-3" /></a>
        </div>
        <p class="rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs text-amber-800 leading-6">
          本部署包含模型管理定制功能。升级时需要合并主仓库更新并重新构建，保留账号、配置和模型设置；直接替换为官方镜像会失去定制界面。此按钮只检查更新，不会重启服务。
        </p>
      </template>
    </div>
    <template #footer>
      <Button
        :loading="loading"
        @click="check"
      >
        重新检查
      </Button><Button @click="open = false">
        关闭
      </Button>
    </template>
  </Modal>
</template>
