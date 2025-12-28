"""Mixed analyzer."""

from langchain_core.messages import BaseMessage

from crs.agents.analyzer.function_analyzer import FunctionAnalyzer
from crs.base.agent_base import BaseAgent
from crs.base.context import CrsContext
from crs.base.exceptions import TooManyErrorsError
from crs.logger import CRS_LOGGER

log = CRS_LOGGER.getChild(__name__)


class MixedAnalyzer(BaseAgent[CrsContext]):
    """
    Analyze the functions changed in a commit.
    """

    def __init__(self, ctx: CrsContext):
        """
        Inits the mixed analyzer.
        """
        super().__init__(ctx)
        self.lsp_ctx, diff = self.init_lsp(ctx.comp_env.checkout(ctx.subproj_commit))

        log.debug(f"Will do mixed mode on diff:\n{diff}")

        self.changed_functions = self.lsp_ctx.lang_config.editor.get_symbols_occurring_in_diff(
            diff, ctx.cp_path_abs / ctx.src_path_rel, functions_only=True
        )

    def run(self) -> CrsContext:
        """Run the agent."""

        log.info("Entering mixed-mode analyzer (agent)...")

        log.debug(
            f"Will analyze the following {len(self.changed_functions)} "
            f"functions: {[f.name for f in self.changed_functions]}..."
        )

        exceptions: list[Exception] = []

        for function in self.changed_functions:
            log.info(f"Mixed mode: Analzying function {function.name} with definition:\n{function.definition}")
            try:
                subagent = FunctionAnalyzer(self.ctx, target_override=(self.lsp_ctx, function.definition))
                # change agent name to prevent overwriting of .md files
                # (TODO: Pylint doesn't like subagent.NAME and we can't disable the warning.)
                setattr(subagent, "NAME", f"analyzer_agent_{function.name}")
                self.ctx = subagent.run()
            # pylint: disable=broad-except
            except Exception as e:
                exceptions.append(e)
                log.critical(f"Mixed-mode analyzer failed for function {function.name}: {e}")

        if exceptions:
            raise TooManyErrorsError(exceptions) if len(exceptions) > 1 else exceptions[0]

        log.info("Mixed-mode analyzer done!")

        return self.ctx

    def _get_initial_messages(self) -> list[BaseMessage]:
        """The messages the LLM shall initially be prompted with."""
        raise NotImplementedError("This should not happen!")
