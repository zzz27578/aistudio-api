<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { ChevronDown, RefreshCw, Search, Boxes, Copy } from 'lucide-vue-next'
import Button from '@/components/ui/Button.vue'
import { modelsApi, type CatalogModel, type CatalogResponse, type ThinkingLevel } from '@/api/models.ts'
import { useToastStore } from '@/stores/toast.ts'

const toast = useToastStore()
const catalog = ref<CatalogResponse | null>(null)
const loading = ref(false)
const error = ref('')
const query = ref('')
const filter = ref('all')
const expanded = ref(new Set<string>())
const saving = ref(new Set<string>())
const dirty = ref(new Set<string>())
const levelLabels: Record<ThinkingLevel, string> = { MINIMAL: '极低', LOW: '低', MEDIUM: '中', HIGH: '高' }
const visibleModels = computed(() => (catalog.value?.models ?? []).filter(model => {
  const matches = `${model.id} ${model.metadata.displayName ?? ''}`.toLowerCase().includes(query.value.toLowerCase())
  return matches && (filter.value === 'all' || (filter.value === 'enabled' ? model.policy.enabled : !model.policy.enabled))
}))
const enabledCount = computed(() => catalog.value?.models.filter(m => m.policy.enabled).length ?? 0)
const publicCount = computed(() => catalog.value?.models.reduce((sum, m) => sum + (m.policy.enabled ?
  Number(m.policy.visible) + m.supported_levels.filter(l => m.policy.thinking_aliases[l]?.enabled && m.policy.thinking_aliases[l]?.visible).length : 0), 0) ?? 0)
const refreshedAt = computed(() => catalog.value?.refreshed_at ? new Date(catalog.value.refreshed_at * 1000).toLocaleString() : '尚未拉取')

async function fetchModels(refresh = false) {
  loading.value = true
  error.value = ''
  try {
    catalog.value = await modelsApi.list(refresh)
    for (const model of catalog.value.models) {
      for (const level of model.supported_levels) {
        model.policy.thinking_aliases[level] ??= { enabled: false, visible: true }
      }
    }
    dirty.value.clear()
    if (refresh) toast.success(catalog.value.refresh_error ? '列表已加载，请查看拉取提示' : '已重新拉取模型，原有设置已保留')
  } catch (e) {
    error.value = e instanceof Error ? e.message : '模型加载失败'
  } finally { loading.value = false }
}
function toggle(id: string) {
  if (expanded.value.has(id)) expanded.value.delete(id)
  else expanded.value.add(id)
}
async function save(model: CatalogModel) {
  saving.value.add(model.id)
  try {
    await modelsApi.save(model.id, model.policy)
    dirty.value.delete(model.id)
    toast.success(`${model.id} 的设置已保存`)
  } catch (e) { toast.error(e instanceof Error ? e.message : '保存失败') }
  finally { saving.value.delete(model.id) }
}
async function copy(value: string) {
  try { await navigator.clipboard.writeText(value); toast.success('模型名称已复制') }
  catch { toast.info('请选中模型名称复制') }
}
onMounted(() => fetchModels())
</script>

<template>
  <div class="space-y-6">
    <div class="flex flex-wrap items-start justify-between gap-4">
      <div>
        <h1 class="text-xl font-semibold text-gray-900">
          模型管理
        </h1>
        <p class="mt-1 text-sm text-gray-500">
          管理模型的启用状态、Key 拉取范围和思考等级后缀。
        </p>
      </div>
      <Button
        :loading="loading"
        :disabled="dirty.size > 0"
        @click="fetchModels(true)"
      >
        <RefreshCw
          v-if="!loading"
          class="w-4 h-4"
        />重新拉取模型
      </Button>
    </div>

    <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
      <div class="rounded-2xl border border-gray-200 bg-white p-5">
        <p class="text-xs text-gray-500">
          已发现模型
        </p><p class="mt-2 text-2xl font-semibold text-gray-900">
          {{ catalog?.models.length ?? '—' }}
        </p>
      </div>
      <div class="rounded-2xl border border-gray-200 bg-white p-5">
        <p class="text-xs text-gray-500">
          已启用
        </p><p class="mt-2 text-2xl font-semibold text-gray-900">
          {{ enabledCount }}
        </p>
      </div>
      <div class="rounded-2xl border border-gray-200 bg-white p-5">
        <p class="text-xs text-gray-500">
          Key 可拉取（含思考后缀）
        </p><p class="mt-2 text-2xl font-semibold text-brand-600">
          {{ publicCount }}
        </p>
      </div>
    </div>

    <div class="rounded-xl border border-brand-100 bg-brand-50 p-4 text-sm text-brand-800 leading-6">
      启用思考等级后缀并保存后，在 New API 渠道重新拉取模型即可选择。后缀会固定思考等级，覆盖请求中的思考强度。
      “Key 可拉取”对所有 Key 生效；隐藏模型仍可通过完整名称调用，关闭“启用模型”才会拒绝调用。
    </div>
    <p
      v-if="catalog?.refresh_error"
      role="status"
      class="rounded-xl bg-amber-50 border border-amber-200 p-4 text-sm text-amber-800"
    >
      {{ catalog.refresh_error }}
    </p>
    <div class="flex flex-wrap gap-3 items-center">
      <div class="relative flex-1 min-w-48">
        <Search class="absolute left-3 top-3 w-4 h-4 text-gray-400" />
        <input
          v-model="query"
          aria-label="搜索模型"
          placeholder="搜索模型名称…"
          class="w-full pl-10 pr-3 py-2.5 bg-white border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-brand-400/40"
        >
      </div>
      <select
        v-model="filter"
        aria-label="按启用状态筛选"
        class="rounded-xl border border-gray-200 bg-white py-2.5 px-3 text-sm"
      >
        <option value="all">
          全部状态
        </option><option value="enabled">
          已启用
        </option><option value="disabled">
          已关闭
        </option>
      </select>
      <span class="text-xs text-gray-500">{{ catalog?.source === 'upstream' ? '上游模型' : '内置列表' }} · {{ refreshedAt }}</span>
    </div>
    <p
      v-if="dirty.size"
      class="text-sm text-amber-700"
      role="status"
    >
      有 {{ dirty.size }} 个模型尚未保存，请分别保存后再重新拉取。
    </p>
    <div
      v-if="error"
      role="alert"
      class="rounded-xl bg-rose-50 p-4 text-sm text-rose-700 flex items-center justify-between gap-3"
    >
      {{ error }}<Button
        size="sm"
        @click="fetchModels()"
      >
        重试
      </Button>
    </div>
    <div
      v-if="loading && !catalog"
      role="status"
      class="py-12 text-center text-gray-500"
    >
      正在加载模型…
    </div>
    <div
      v-else-if="catalog && !visibleModels.length"
      class="py-12 text-center text-gray-500"
    >
      <Boxes class="mx-auto mb-3 w-8 h-8 text-gray-300" />没有匹配的模型
    </div>

    <div class="space-y-3">
      <section
        v-for="model in visibleModels"
        :key="model.id"
        class="rounded-2xl bg-white border border-gray-200 overflow-hidden"
      >
        <button
          type="button"
          :aria-expanded="expanded.has(model.id)"
          :aria-controls="`panel-${model.id}`"
          class="w-full flex items-center gap-3 p-4 sm:p-5 text-left cursor-pointer hover:bg-gray-50 transition-colors focus-visible:outline-2 focus-visible:outline-brand-500"
          @click="toggle(model.id)"
        >
          <Boxes class="w-5 h-5 text-gray-400 shrink-0" />
          <div class="min-w-0 flex-1">
            <h2 class="font-medium text-sm text-gray-900 break-all">
              {{ model.id }}
            </h2>
            <p class="mt-1 text-xs text-gray-500">
              {{ model.metadata.displayName ?? model.metadata.category ?? '上游模型' }}
            </p>
          </div>
          <span
            class="hidden sm:inline text-xs rounded-lg px-2 py-1"
            :class="model.policy.enabled ? 'text-emerald-700 bg-emerald-50' : 'text-gray-500 bg-gray-100'"
          >{{ model.policy.enabled ? '已启用' : '已关闭' }}</span>
          <span class="hidden md:inline text-xs text-gray-500">{{ model.policy.visible ? 'Key 可见' : 'Key 隐藏' }}</span>
          <span
            v-if="dirty.has(model.id)"
            class="text-xs text-amber-600"
          >未保存</span>
          <ChevronDown
            class="w-4 h-4 text-gray-400 shrink-0 transition-transform"
            :class="expanded.has(model.id) ? 'rotate-180' : ''"
          />
        </button>
        <div
          v-if="expanded.has(model.id)"
          :id="`panel-${model.id}`"
          class="border-t border-gray-100 p-4 sm:p-5 space-y-5"
        >
          <p
            v-if="model.metadata.description"
            class="text-xs text-gray-500 leading-5"
          >
            {{ model.metadata.description }}
          </p>
          <fieldset
            :disabled="saving.has(model.id)"
            class="space-y-5 disabled:opacity-60"
            @change="dirty.add(model.id)"
          >
            <div class="grid sm:grid-cols-2 gap-4">
              <label class="flex items-start gap-3 cursor-pointer rounded-xl bg-gray-50 p-4">
                <input
                  v-model="model.policy.enabled"
                  type="checkbox"
                  class="mt-1 w-4 h-4 accent-brand-500"
                >
                <span><span class="block text-sm font-medium text-gray-800">启用模型</span><span class="block mt-1 text-xs text-gray-500">关闭后拒绝该模型及其思考后缀的调用</span></span>
              </label>
              <label class="flex items-start gap-3 cursor-pointer rounded-xl bg-gray-50 p-4">
                <input
                  v-model="model.policy.visible"
                  type="checkbox"
                  class="mt-1 w-4 h-4 accent-brand-500"
                >
                <span><span class="block text-sm font-medium text-gray-800">Key 可拉取基础模型</span><span class="block mt-1 text-xs text-gray-500">控制基础名称是否出现在模型列表中</span></span>
              </label>
            </div>
            <div>
              <h3 class="text-sm font-medium text-gray-800">
                思考等级后缀
              </h3>
              <p class="mt-1 text-xs text-gray-500">
                仅提供上游明确声明的等级。基础名称仍使用原有配置或请求参数。
              </p>
              <div
                v-if="model.supported_levels.length"
                class="mt-3 space-y-2"
              >
                <div
                  v-for="level in model.supported_levels"
                  :key="level"
                  class="rounded-xl border border-gray-200 p-3 flex flex-wrap items-center gap-3"
                >
                  <span class="text-xs font-medium text-brand-700 w-16">{{ levelLabels[level] }} {{ level }}</span>
                  <code class="text-xs text-gray-600 break-all flex-1 min-w-40 select-all">{{ model.aliases[level] }}</code>
                  <button
                    type="button"
                    :aria-label="`复制 ${model.aliases[level]}`"
                    class="p-1.5 rounded-lg text-gray-400 hover:bg-gray-100 cursor-pointer"
                    @click="copy(model.aliases[level])"
                  >
                    <Copy class="w-3.5 h-3.5" />
                  </button>
                  <label class="text-xs flex items-center gap-2 cursor-pointer"><input
                    v-model="model.policy.thinking_aliases[level]!.enabled"
                    type="checkbox"
                    class="w-4 h-4 accent-brand-500"
                  >启用后缀</label>
                  <label class="text-xs flex items-center gap-2 cursor-pointer"><input
                    v-model="model.policy.thinking_aliases[level]!.visible"
                    type="checkbox"
                    class="w-4 h-4 accent-brand-500"
                  >Key 可拉取</label>
                </div>
              </div>
              <p
                v-else
                class="mt-3 text-xs text-gray-500 rounded-xl bg-gray-50 p-3"
              >
                此模型未声明思考等级，无法配置等级后缀。
              </p>
            </div>
          </fieldset>
          <div class="flex flex-wrap justify-between items-center gap-3">
            <span class="text-xs text-gray-400">设置保存后立即生效，重新拉取不会覆盖设置。</span>
            <Button
              variant="primary"
              size="sm"
              :loading="saving.has(model.id)"
              :disabled="!dirty.has(model.id)"
              @click="save(model)"
            >
              保存模型设置
            </Button>
          </div>
        </div>
      </section>
    </div>
  </div>
</template>
