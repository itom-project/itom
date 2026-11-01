#include "gtest/gtest.h"

#include "../../Qitom/llm/contextAssemblyService.h"
#include "../../Qitom/llm/promptPipeline.h"
#include "../../Qitom/llm/llmSecurityPolicy.h"
#include "../../Qitom/llm/llmLogRedaction.h"
#include "../../Qitom/llm/workspaceApplyService.h"
#include "../../Qitom/llm/workspaceChangeSet.h"

#include <QDir>
#include <QFile>
#include <QTemporaryDir>
#include <QTextStream>

using namespace ito::llm;

TEST(LlmCoreTest, ContextAssemblyBuildsMultiFileContext)
{
    ContextAssemblyInput input;
    input.selectedText = "print('selected')";
    input.activeFilePath = "active.py";
    input.activeFileContent = "print('active')";

    ContextFileContent file1;
    file1.filePath = "a.py";
    file1.content = "print('a')";

    ContextFileContent file2;
    file2.filePath = "b.py";
    file2.content = QString(3000, 'x');

    input.additionalFiles << file1 << file2;

    ContextAssemblyService svc;
    ContextEnvelope env = svc.build(input);

    EXPECT_EQ(env.activeFilePath, QString("active.py"));
    EXPECT_EQ(env.selectedText, QString("print('selected')"));
    EXPECT_TRUE(env.additionalFiles.size() >= 1);
    EXPECT_TRUE(env.referencedFilePaths.contains("a.py"));
    EXPECT_GT(env.estimatedTokenCount, 0);
}

TEST(LlmCoreTest, PromptPipelineIncludesAdditionalFilesAndTruncation)
{
    ContextEnvelope env;
    env.selectedText = "x = 1";
    env.activeFilePath = "active.py";
    env.activeFileContent = "print(x)";

    ContextFileContent additional;
    additional.filePath = "helper.py";
    additional.content = "def f():\n    return 1";
    env.additionalFiles << additional;
    env.referencedFilePaths << additional.filePath;
    env.truncationReport = "activeFileContent truncated";
    env.estimatedTokenCount = 42;

    PromptPipeline pipeline;
    QList<LlmMessage> messages =
        pipeline.buildMessages(LlmTaskType::Generate, env, "Generate code");

    ASSERT_GT(messages.size(), 3);

    bool hasAdditional = false;
    bool hasTruncation = false;
    bool hasEstimate = false;

    for (int i = 0; i < messages.size(); ++i)
    {
        hasAdditional = hasAdditional || messages[i].content.contains("Additional file context (helper.py)");
        hasTruncation = hasTruncation || messages[i].content.contains("Context truncation report");
        hasEstimate = hasEstimate || messages[i].content.contains("Estimated context tokens: 42");
    }

    EXPECT_TRUE(hasAdditional);
    EXPECT_TRUE(hasTruncation);
    EXPECT_TRUE(hasEstimate);
}

TEST(LlmCoreTest, SecurityPolicyBlocksDisabledAndValidatesCloud)
{
    LlmSecurityPolicy policy;

    LlmSettings disabledSettings;
    disabledSettings.enabled = false;
    disabledSettings.mode = LlmMode::Disabled;

    LlmRequest req;
    req.mode = LlmMode::Disabled;

    LlmError err;
    ito::RetVal retval = policy.validateRequest(disabledSettings, req, &err);
    EXPECT_TRUE(retval.containsError());
    EXPECT_EQ(err.category, LlmErrorCategory::Policy);

    LlmSettings cloudSettings;
    cloudSettings.enabled = true;
    cloudSettings.mode = LlmMode::Cloud;
    cloudSettings.provider = "github-models";
    cloudSettings.model = "gpt-4.1";
    cloudSettings.endpoint = "https://models.inference.ai.azure.com";
    cloudSettings.apiKey = "secret-token";

    req.mode = LlmMode::Cloud;
    req.apiKey = "secret-token";

    retval = policy.validateRequest(cloudSettings, req, &err);
    EXPECT_FALSE(retval.containsError());
}

TEST(LlmCoreTest, WorkspaceChangeSetParserParsesOperations)
{
    const QString text =
        "<<<LLM_CHANGE create path=a.py>>>\nprint('a')\n<<<END_LLM_CHANGE>>>\n"
        "<<<LLM_CHANGE update path=b.py>>>\nprint('b')\n<<<END_LLM_CHANGE>>>\n"
        "<<<LLM_CHANGE delete path=c.py>>>\n<<<END_LLM_CHANGE>>>";

    WorkspaceChangeSetParser parser;
    WorkspaceChangeSet set = parser.parse(text);

    ASSERT_EQ(set.changes.size(), 3);
    EXPECT_EQ(set.changes[0].type, WorkspaceChangeType::CreateFile);
    EXPECT_EQ(set.changes[1].type, WorkspaceChangeType::UpdateFile);
    EXPECT_EQ(set.changes[2].type, WorkspaceChangeType::DeleteFile);
    EXPECT_EQ(set.changes[2].content, QString());
}

TEST(LlmCoreTest, WorkspacePolicyRequiresConfirmationAndBlocksAutoApply)
{
    LlmSecurityPolicy policy;

    WorkspaceChangeSet set;
    WorkspaceChange del;
    del.type = WorkspaceChangeType::DeleteFile;
    del.filePath = "obsolete.py";
    set.changes << del;

    WorkspacePolicyDecision d1 = policy.evaluateWorkspaceChangeSet(set, false);
    EXPECT_TRUE(d1.allowed);
    EXPECT_TRUE(d1.requiresConfirmation);

    WorkspacePolicyDecision d2 = policy.evaluateWorkspaceChangeSet(set, true);
    EXPECT_FALSE(d2.allowed);
    EXPECT_TRUE(d2.requiresConfirmation);
}

TEST(LlmCoreTest, WorkspacePolicyAllowsLowRiskAutoApply)
{
    LlmSecurityPolicy policy;

    WorkspaceChangeSet set;
    WorkspaceChange upd;
    upd.type = WorkspaceChangeType::UpdateFile;
    upd.filePath = "script.py";
    upd.content = "print('ok')";
    set.changes << upd;

    WorkspacePolicyDecision d = policy.evaluateWorkspaceChangeSet(set, true);
    EXPECT_TRUE(d.allowed);
    EXPECT_FALSE(d.requiresConfirmation);
}

TEST(LlmCoreTest, WorkspaceApplyServiceCreatesUpdatesAndDeletesFiles)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString root = tempDir.path();
    const QString existingPath = QDir(root).filePath("existing.py");

    {
        QFile initial(existingPath);
        ASSERT_TRUE(initial.open(QIODevice::WriteOnly | QIODevice::Text));
        QTextStream s(&initial);
        s << "print('old')";
    }

    WorkspaceChangeSet set;

    WorkspaceChange create;
    create.type = WorkspaceChangeType::CreateFile;
    create.filePath = "new.py";
    create.content = "print('new')";
    set.changes << create;

    WorkspaceChange update;
    update.type = WorkspaceChangeType::UpdateFile;
    update.filePath = "existing.py";
    update.content = "print('updated')";
    set.changes << update;

    WorkspaceChange del;
    del.type = WorkspaceChangeType::DeleteFile;
    del.filePath = "new.py";
    set.changes << del;

    WorkspaceApplyService svc;
    WorkspaceApplyResult result;
    ito::RetVal ret = svc.apply(root, set, &result);

    EXPECT_FALSE(ret.containsError());
    EXPECT_EQ(result.createdFiles, 1);
    EXPECT_EQ(result.updatedFiles, 1);
    EXPECT_EQ(result.deletedFiles, 1);

    QFile existing(existingPath);
    ASSERT_TRUE(existing.open(QIODevice::ReadOnly | QIODevice::Text));
    const QString updatedText = QString::fromUtf8(existing.readAll());
    EXPECT_TRUE(updatedText.contains("updated"));

    EXPECT_FALSE(QFileInfo::exists(QDir(root).filePath("new.py")));
}

TEST(LlmCoreTest, WorkspaceApplyServiceBlocksUnsafePath)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    WorkspaceChangeSet set;
    WorkspaceChange upd;
    upd.type = WorkspaceChangeType::UpdateFile;
    upd.filePath = "..\\outside.py";
    upd.content = "print('x')";
    set.changes << upd;

    WorkspaceApplyService svc;
    WorkspaceApplyResult result;
    ito::RetVal ret = svc.apply(tempDir.path(), set, &result);

    EXPECT_TRUE(ret.containsError());
    ASSERT_FALSE(result.errors.isEmpty());
    EXPECT_TRUE(result.errors.first().contains("Unsafe path blocked"));
}

TEST(LlmCoreTest, WorkspaceChangeSetParserIgnoresMalformedBlocks)
{
    const QString text = QString::fromLatin1(R"(<<<LLM_CHANGE update path=ok.py>>>
print('ok')
<<<END_LLM_CHANGE>>>
<<<LLM_CHANGE update>>>
no path
<<<END_LLM_CHANGE>>>)");

    WorkspaceChangeSetParser parser;
    WorkspaceChangeSet set = parser.parse(text);

    ASSERT_EQ(set.changes.size(), 1);
    EXPECT_EQ(set.changes[0].filePath, QString("ok.py"));
}

TEST(LlmCoreTest, LogRedactionMasksMultipleSecretPatterns)
{
    const QString in =
        "Authorization: Bearer abc123\n"
        "Authorization: Basic dXNlcjpwYXNz\n"
        "apiKey=xyz\n"
        "password: p@ss\n"
        "secret = hidden";

    const QString out = LlmLogRedaction::redactAll(in);

    EXPECT_FALSE(out.contains("abc123"));
    EXPECT_FALSE(out.contains("dXNlcjpwYXNz"));
    EXPECT_FALSE(out.contains("xyz"));
    EXPECT_FALSE(out.contains("p@ss"));
    EXPECT_FALSE(out.contains("hidden"));
    EXPECT_TRUE(out.contains("***"));
}
