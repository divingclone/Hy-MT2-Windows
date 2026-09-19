<script lang="ts">
  let { text }: { text: string } = $props();
  let anchor: HTMLSpanElement;
  let content: HTMLSpanElement;
  let top = $state(0);
  let left = $state(0);
  function place() {
    if (!anchor || !content) return;
    const rect = anchor.getBoundingClientRect();
    left = Math.max(
      12,
      Math.min(rect.left - 24, window.innerWidth - content.offsetWidth - 12),
    );
    top =
      rect.bottom + 8 + content.offsetHeight <= window.innerHeight - 12
        ? rect.bottom + 8
        : Math.max(12, rect.top - content.offsetHeight - 8);
  }
</script>

<svelte:window onresize={place} onscroll={place} />

<span
  class="help"
  tabindex="0"
  role="button"
  aria-label={text}
  bind:this={anchor}
  onmouseenter={place}
  onfocus={place}
  onclick={(event) => {
    event.preventDefault();
    event.currentTarget.focus();
  }}
  onkeydown={(event) => {
    if (event.key === " " || event.key === "Enter") event.preventDefault();
  }}
  >?
  <span
    class="help-content"
    role="tooltip"
    bind:this={content}
    style:top={`${top}px`}
    style:left={`${left}px`}>{text}</span
  >
</span>

<style>
  .help {
    display: inline-flex;
    position: relative;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    margin-left: 5px;
    border: 1px solid #aab9ad;
    border-radius: 50%;
    font-size: 11px;
    font-weight: 500;
    color: #64776a;
    cursor: help;
    vertical-align: middle;
  }
  .help-content {
    visibility: hidden;
    pointer-events: none;
    position: fixed;
    width: min(310px, 65vw);
    padding: 12px 14px;
    border-radius: 8px;
    background: #233d32;
    color: #fff;
    font-size: 12px;
    font-weight: 400;
    line-height: 1.7;
    box-shadow: 0 6px 24px #12291d26;
    z-index: 20;
    white-space: normal;
  }
  .help:hover .help-content,
  .help:focus .help-content {
    visibility: visible;
  }
</style>
